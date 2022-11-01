import numpy as np
import math
import torch
import torch.nn.functional as F
from typing import List, Mapping, Optional, Text, Tuple, Union

def pad_poses(p: np.ndarray) -> np.ndarray:
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
  return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[..., :3, :4]

def transform_poses_pca(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Transforms poses so principal components lie on XYZ axes.
  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.
  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
  t = poses[:, :3, 3]
  t_mean = t.mean(axis=0)
  t = t - t_mean

  eigval, eigvec = np.linalg.eig(t.T @ t)
  # Sort eigenvectors in order of largest to smallest eigenvalue.
  inds = np.argsort(eigval)[::-1]
  eigvec = eigvec[:, inds]
  rot = eigvec.T
  if np.linalg.det(rot) < 0:
    rot = np.diag(np.array([1, 1, -1])) @ rot

  transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
  poses_recentered = unpad_poses(transform @ pad_poses(poses))
  transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

  # Flip coordinate system if z component of y-axis is negative
  if poses_recentered.mean(axis=0)[2, 1] < 0:
    poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
    transform = np.diag(np.array([1, -1, -1, 1])) @ transform

  # Just make sure it's it in the [-1, 1]^3 cube
  scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
  poses_recentered[:, :3, 3] *= scale_factor
  transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

  return poses_recentered, transform, scale_factor

def focus_point_fn(poses: np.ndarray) -> np.ndarray:
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt

def normalize(x: np.ndarray) -> np.ndarray:
  """Normalization helper function."""
  return x / np.linalg.norm(x)

def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
  """Construct lookat view matrix."""
  vec2 = normalize(lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m

def integrate_weights(w):
  """Compute the cumulative sum of w, assuming all weight vectors sum to 1.
  The output's size on the last dimension is one greater than that of the input,
  because we're computing the integral corresponding to the endpoints of a step
  function, not the integral of the interior/bin values.
  Args:
    w: Tensor, which will be integrated along the last axis. This is assumed to
      sum to 1 along the last axis, and this function will (silently) break if
      that is not the case.
  Returns:
    cw0: Tensor, the integral of w, where cw0[..., 0] = 0 and cw0[..., -1] = 1
  """
  cw = torch.clip(torch.cumsum(w[..., :-1], axis=-1), max=1)
  shape = cw.shape[:-1] + (1,)
  # Ensure that the CDF starts with exactly 0 and ends with exactly 1.
  cw0 = torch.cat([torch.zeros(shape), cw, torch.ones(shape)], axis=-1)
  return cw0

# def interp(*args):
#   """A gather-based (GPU-friendly) vectorized replacement for jnp.interp()."""
#   args_flat = [x.reshape([-1, x.shape[-1]]) for x in args]
#   import pdb
#   pdb.set_trace()
#   ret = np.vmap(np.interp())(*args_flat).reshape(args[0].shape)
#   return ret

def invert_cdf(u, t, w_logits, use_gpu_resampling=False):
  """Invert the CDF defined by (t, w) at the points specified by u in [0, 1)."""
  # Compute the PDF and CDF for each weight vector.
  w = F.softmax(w_logits, dim=-1)
  cw = integrate_weights(w)
  # Interpolate into the inverse CDF.
  # interp_fn = math.interp
  t_new = np.interp(u, cw, t)
  return t_new

def sample(rng,
           t,
           w_logits,
           num_samples,
           single_jitter=False,
           deterministic_center=False,
           use_gpu_resampling=False):
  """Piecewise-Constant PDF sampling from a step function.
  Args:
    rng: random number generator (or None for `linspace` sampling).
    t: [..., num_bins + 1], bin endpoint coordinates (must be sorted)
    w_logits: [..., num_bins], logits corresponding to bin weights
    num_samples: int, the number of samples.
    single_jitter: bool, if True, jitter every sample along each ray by the same
      amount in the inverse CDF. Otherwise, jitter each sample independently.
    deterministic_center: bool, if False, when `rng` is None return samples that
      linspace the entire PDF. If True, skip the front and back of the linspace
      so that the centers of each PDF interval are returned.
    use_gpu_resampling: bool, If True this resamples the rays based on a
      "gather" instruction, which is fast on GPUs but slow on TPUs. If False,
      this resamples the rays based on brute-force searches, which is fast on
      TPUs, but slow on GPUs.
  Returns:
    t_samples: jnp.ndarray(float32), [batch_size, num_samples].
  """
  eps = 1e-9 
  
  # Draw uniform samples.
  if rng is None:
    # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
    if deterministic_center:
      pad = 1 / (2 * num_samples)
      u = torch.linspace(pad, 1. - pad - eps, num_samples)
    else:
      u = torch.linspace(0, 1. - eps, num_samples)
    u = torch.broadcast_to(u, t.shape[:-1] + (num_samples,))
    
    return invert_cdf(u, torch.from_numpy(t), torch.from_numpy(w_logits))

def generate_ellipse_path(poses: np.ndarray,
                          n_frames: int = 120,
                          const_speed: bool = True,
                          z_variation: float = 0.,
                          z_phase: float = 0.) -> np.ndarray:
  """Generate an elliptical render path based on the given poses."""
  # Calculate the focal point for the path (cameras point toward this).
  center = focus_point_fn(poses)
  # Path height sits at z=0 (in middle of zero-mean capture pattern).
  offset = np.array([center[0], center[1], 0])

  # Calculate scaling for ellipse axes based on input camera positions.
  sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
  # Use ellipse that is symmetric about the focal point in xy.
  low = -sc + offset
  high = sc + offset
  # Optional height variation need not be symmetric
  z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
  z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

  def get_positions(theta):
    # Interpolate between bounds with trig functions to get ellipse in x-y.
    # Optionally also interpolate in z to change camera height along path.
    return np.stack([
        low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
        low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
        z_variation * (z_low[2] + (z_high - z_low)[2] *
                       (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
    ], -1)

  theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
  positions = get_positions(theta)

  if const_speed:
    # Resample theta angles so that the velocity is closer to constant.
    lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    theta = sample(None, theta, np.log(lengths), n_frames + 1)
    positions = get_positions(theta)

  # Throw away duplicated last position.
  positions = positions[:-1]

  # Set path's up vector to axis closest to average of input pose up vectors.
  avg_up = poses[:, :3, 1].mean(0)
  avg_up = avg_up / np.linalg.norm(avg_up)
  ind_up = np.argmax(np.abs(avg_up))
  up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

  return np.stack([viewmatrix(p - center, up, p) for p in positions])

def gen_renderposes(poses, bounds, n_frames):

    # Separate out 360 versus forward facing scenes.

    poses = np.concatenate([poses[..., 1:2], -poses[..., 0:1], poses[..., 2:]], -1)
    poses_transformed, transform, scale_factor = transform_poses_pca(poses)
    pposes = pad_poses(poses_transformed)
    # Automatically generated inward-facing elliptical render path.
    render_poses = generate_ellipse_path(
        poses_transformed,
        n_frames=n_frames,
        z_variation=0,
        z_phase=0)
    transform_inv = np.linalg.inv(transform)
    render_poses = np.stack([transform_inv @ pose for pose in pad_poses(render_poses)], axis=0)
    render_poses[:, :3, :3] *= scale_factor
    return unpad_poses(render_poses)
