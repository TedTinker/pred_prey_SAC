import numpy as np
import torch
import gin

from collections import namedtuple
from utils import device, get_free_mem, delete_these


RecurrentBatch = namedtuple('RecurrentBatch', 'o s e a r d m')

def as_probas(positive_values: np.array) -> np.array:
    return positive_values / np.sum(positive_values)

def as_tensor_on_device(np_array: np.array):
    return torch.tensor(np_array).float().to(device)

@gin.configurable(module=__name__)
class RecurrentReplayBuffer:

    """Use this version when num_bptt == max_episode_len"""
    
    def __init__(
        self,
        o_dim = (16,16,4),
        a_dim = 2,
        max_episode_len = 501,  # this will also serve as num_bptt
        segment_len=None,  # for non-overlapping truncated bptt, maybe need a large batch size
    ):
        # placeholders
    
        capacity = 100
      
        self.o = np.zeros((capacity, max_episode_len + 1) + o_dim, dtype='float32')
        self.s = np.zeros((capacity, max_episode_len + 1, 1), dtype='float32')
        self.e = np.zeros((capacity, max_episode_len + 1, 1), dtype='float32')
        self.a = np.zeros((capacity, max_episode_len, a_dim), dtype='float32')
        self.r = np.zeros((capacity, max_episode_len, 1), dtype='float32')
        self.d = np.zeros((capacity, max_episode_len, 1), dtype='float32')
        self.m = np.zeros((capacity, max_episode_len, 1), dtype='float32')
        self.ep_len = np.zeros((capacity,), dtype='float32')
        self.ready_for_sampling = np.zeros((capacity,), dtype='float32')
      
        # pointers
      
        self.episode_ptr = 0
        self.time_ptr = 0
      
        # trackers
      
        self.starting_new_episode = True
        self.num_episodes = 0
      
        # hyper-parameters
      
        self.capacity = capacity
        self.o_dim = o_dim
        self.a_dim = a_dim
      
        self.max_episode_len = max_episode_len
      
        if segment_len is not None:
            assert max_episode_len % segment_len == 0  # e.g., if max_episode_len = 1000, then segment_len = 100 is ok
      
        self.segment_len = segment_len

    def push(self, p):
        
        o, s, e, a, r, no, ns, ne, d, cutoff = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]
        
        # zero-out current slot at the beginning of an episode
      
        if self.starting_new_episode:
            self.o[self.episode_ptr] = 0
            self.s[self.episode_ptr] = 0
            self.e[self.episode_ptr] = 0
            self.a[self.episode_ptr] = 0
            self.r[self.episode_ptr] = 0
            self.d[self.episode_ptr] = 0
            self.m[self.episode_ptr] = 0
            self.ep_len[self.episode_ptr] = 0
            self.ready_for_sampling[self.episode_ptr] = 0
            self.starting_new_episode = False
      
        # fill placeholders
        
        self.o[self.episode_ptr, self.time_ptr] = o
        self.s[self.episode_ptr, self.time_ptr] = s
        self.e[self.episode_ptr, self.time_ptr] = e
        self.a[self.episode_ptr, self.time_ptr] = a
        self.r[self.episode_ptr, self.time_ptr] = r
        self.d[self.episode_ptr, self.time_ptr] = d
        self.m[self.episode_ptr, self.time_ptr] = 1
        self.ep_len[self.episode_ptr] += 1
      
        if d or cutoff:
      
            # fill placeholders
        
            self.o[self.episode_ptr, self.time_ptr+1] = no
            self.s[self.episode_ptr, self.time_ptr+1] = ns
            self.e[self.episode_ptr, self.time_ptr+1] = ne
            self.ready_for_sampling[self.episode_ptr] = 1
        
            # reset pointers
        
            self.episode_ptr = (self.episode_ptr + 1) % self.capacity
            self.time_ptr = 0
        
            # update trackers
        
            self.starting_new_episode = True
            if self.num_episodes < self.capacity:
                self.num_episodes += 1
      
        else:
      
            # update pointers
        
              self.time_ptr += 1
    
    def sample(self, batch_size = 16):
      
        if(self.num_episodes < batch_size): return False
      
        # sample episode indices
      
        options = np.where(self.ready_for_sampling == 1)[0]
        ep_lens_of_options = self.ep_len[options]
        probas_of_options = as_probas(ep_lens_of_options)
        choices = np.random.choice(options, p=probas_of_options, size=batch_size)
      
        ep_lens_of_choices = self.ep_len[choices]
      
        if self.segment_len is None:
      
            # grab the corresponding numpy array
            # and save computational effort for lstm
        
            max_ep_len_in_batch = int(np.max(ep_lens_of_choices))
        
            o = self.o[choices][:, :max_ep_len_in_batch+1, :]
            s = self.s[choices][:, :max_ep_len_in_batch+1, :]
            e = self.e[choices][:, :max_ep_len_in_batch+1, :]
            a = self.a[choices][:, :max_ep_len_in_batch, :]
            r = self.r[choices][:, :max_ep_len_in_batch, :]
            d = self.d[choices][:, :max_ep_len_in_batch, :]
            m = self.m[choices][:, :max_ep_len_in_batch, :]
        
            # convert to tensors on the right device
        
            o = as_tensor_on_device(o).view((batch_size, max_ep_len_in_batch+1) + self.o_dim)
            s = as_tensor_on_device(s).view(batch_size, max_ep_len_in_batch+1, 1)
            e = as_tensor_on_device(e).view(batch_size, max_ep_len_in_batch+1, 1)
            a = as_tensor_on_device(a).view(batch_size, max_ep_len_in_batch, self.a_dim)
            r = as_tensor_on_device(r).view(batch_size, max_ep_len_in_batch, 1)
            d = as_tensor_on_device(d).view(batch_size, max_ep_len_in_batch, 1)
            m = as_tensor_on_device(m).view(batch_size, max_ep_len_in_batch, 1)
            return RecurrentBatch(o, s, e, a, r, d, m)
      
        else:
      
            num_segments_for_each_item = np.ceil(ep_lens_of_choices / self.segment_len).astype(int)
        
            o = self.o[choices]
            s = self.s[choices]
            e = self.e[choices]
            a = self.a[choices]
            r = self.r[choices]
            d = self.d[choices]
            m = self.m[choices]
        
            o_seg = np.zeros((batch_size, self.segment_len + 1) + self.o_dim)
            s_seg = np.zeros((batch_size, self.segment_len + 1, 1))
            e_seg = np.zeros((batch_size, self.segment_len + 1, 1))
            a_seg = np.zeros((batch_size, self.segment_len, self.a_dim))
            r_seg = np.zeros((batch_size, self.segment_len, 1))
            d_seg = np.zeros((batch_size, self.segment_len, 1))
            m_seg = np.zeros((batch_size, self.segment_len, 1))
        
            for i in range(batch_size):
                start_idx = np.random.randint(num_segments_for_each_item[i]) * self.segment_len
                o_seg[i] = o[i][start_idx:start_idx + self.segment_len + 1]
                s_seg[i] = s[i][start_idx:start_idx + self.segment_len + 1]
                e_seg[i] = e[i][start_idx:start_idx + self.segment_len + 1]
                a_seg[i] = a[i][start_idx:start_idx + self.segment_len]
                r_seg[i] = r[i][start_idx:start_idx + self.segment_len]
                d_seg[i] = d[i][start_idx:start_idx + self.segment_len]
                m_seg[i] = m[i][start_idx:start_idx + self.segment_len]
        
            o_seg = as_tensor_on_device(o_seg)
            s_seg = as_tensor_on_device(s_seg)
            e_seg = as_tensor_on_device(e_seg)
            a_seg = as_tensor_on_device(a_seg)
            r_seg = as_tensor_on_device(r_seg)
            d_seg = as_tensor_on_device(d_seg)
            m_seg = as_tensor_on_device(m_seg)
            return RecurrentBatch(o_seg, s_seg, e_seg, a_seg, r_seg, d_seg, m_seg)
    