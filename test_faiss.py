# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faiss                     # make faiss available
import numpy as np
#%%
d = 64                           # dimension
nb = 100000                      # database size
nq = 10                      # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.


ngpus = faiss.get_num_gpus()

print("number of GPUs:", ngpus)

cpu_index = faiss.IndexFlatL2(d)

gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
    cpu_index
)
#%%
import face_recognition
import numpy as np
known_image = face_recognition.load_image_file('/home/tmt/Documents/face/collection/TMT/beauty_20190915092027.jpg')

known_face_location = face_recognition.face_locations(known_image, number_of_times_to_upsample=1)

print(known_face_location)
known_encoding = face_recognition.face_encodings(known_image, known_face_locations=known_face_location)[0]
print(type(known_encoding))

print(known_encoding.shape)
known_encoding=np.reshape(known_encoding, (1,128))
#%%
known_image = face_recognition.load_image_file('/home/tmt/Documents/face/collection/Cong Anh/53491942_1224175431066305_4033200567001022464_n.jpg')

known_face_location = face_recognition.face_locations(known_image, number_of_times_to_upsample=1)

print(known_face_location)
known_encoding2 = face_recognition.face_encodings(known_image, known_face_locations=known_face_location)[0]
print(type(known_encoding2))

print(known_encoding2.shape)
known_encoding2=np.reshape(known_encoding2, (1,128))

#%%
known_image = face_recognition.load_image_file('/home/tmt/Documents/face/collection/Khanh Huyen/IMG_20191227_164141.jpg')

known_face_location = face_recognition.face_locations(known_image, number_of_times_to_upsample=1)

print(known_face_location)
known_encoding3 = face_recognition.face_encodings(known_image, known_face_locations=known_face_location)[0]
print(type(known_encoding3))

print(known_encoding3.shape)
known_encoding3=np.reshape(known_encoding3, (1,128))
#%%
known_image = face_recognition.load_image_file('/home/tmt/Documents/face/collection/Truong Sinh/2019-10-30.jpg')

known_face_location = face_recognition.face_locations(known_image, number_of_times_to_upsample=1)

print(known_face_location)
known_encoding5 = face_recognition.face_encodings(known_image, known_face_locations=known_face_location)[0]
print(type(known_encoding5))

print(known_encoding5.shape)
known_encoding5=np.reshape(known_encoding5, (1,128))

#%%
#%%
known_image = face_recognition.load_image_file('/home/tmt/Documents/face/collection/TMT/beauty_20191009001401.jpg')

known_face_location = face_recognition.face_locations(known_image, number_of_times_to_upsample=1)

print(known_face_location)
known_encoding6 = face_recognition.face_encodings(known_image, known_face_locations=known_face_location)[0]
print(type(known_encoding6))

print(known_encoding6.shape)
known_encoding6=np.reshape(known_encoding6, (1,128))





#%%
encoding=np.concatenate((known_encoding,known_encoding2,known_encoding3,known_encoding5,known_encoding6),axis=0)
#%%
known_image = face_recognition.load_image_file('/home/tmt/Documents/face/collection/TMT/beauty_20191009001647.jpg')

known_face_location = face_recognition.face_locations(known_image, number_of_times_to_upsample=1)

print(known_face_location)
known_encoding4 = face_recognition.face_encodings(known_image, known_face_locations=known_face_location)[0]
print(type(known_encoding4))

print(known_encoding4.shape)
known_encoding4=np.reshape(known_encoding4, (1,128))
known_encoding4=known_encoding4.astype(np.float32)
#%%
print(type(encoding))
encoding2=encoding.astype(np.float32)
#%%
import faiss                     # make faiss available
d=128
ngpus = faiss.get_num_gpus()

print("number of GPUs:", ngpus)

cpu_index = faiss.IndexFlatL2(d)

gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
    cpu_index
)
#%%
gpu_index.add(encoding2)              # add vectors to the index
print('index',gpu_index.ntotal)

#%%
k = 2                                       # we want to see 4 nearest neighbors
D, I = gpu_index.search(known_encoding4, k)      # actual search
print(I[0][0])                                # neighbors of the 5 first queries
print(I[-5:])                               # neighbors of the 5 last queries

#%%
known_encoding_faces=[]
known_encoding_faces.append(known_encoding)
known_encoding_faces.append(known_encoding2)
known_encoding_faces.append(known_encoding3)
known_encoding_faces.append(known_encoding5)
known_encoding_faces.append(known_encoding6)
#%%
known_encoding_faces2=np.asarray(known_encoding_faces)
known_encoding_faces2=np.reshape(known_encoding_faces2, (5,128))
known_encoding_faces2=known_encoding_faces2.astype(np.float32)
#%%
known_encoding_faces2.shape
#%%
import faiss                     # make faiss available
d=128
ngpus = faiss.get_num_gpus()

print("number of GPUs:", ngpus)

cpu_index = faiss.IndexFlatL2(d)

gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
    cpu_index
)
#%%
gpu_index.add(known_encoding_faces2)              # add vectors to the index
print('index',gpu_index.ntotal)
#%%
type(gpu_index)
#%%
k = 2                                       # we want to see 4 nearest neighbors
D, I = gpu_index.search(known_encoding4, k)      # actual search
print(I[0][0])                                # neighbors of the 5 first queries
print(I[-5:]) 


#%%
k = 4                        # we want to see 4 nearest neighbors
D, I = gpu_index.search(xq, k)  # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                   # neighbors of the 5 last queries





#%%
gpu_index.add(xb)              # add vectors to the index
print('index',gpu_index.ntotal)
#%%
k = 4                        # we want to see 4 nearest neighbors
D, I = gpu_index.search(xq, k)  # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
#%%
result=xb[381,:]
#%%
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

d = 64                           # dimension
nb = 100000                      # database size
nq = 1                 # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

import faiss                     # make faiss available

res = faiss.StandardGpuResources()  # use a single GPU

## Using a flat index

index_flat = faiss.IndexFlatL2(d)  # build a flat (CPU) index

# make it a flat GPU index
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

gpu_index_flat.add(xb)         # add vectors to the index
print(gpu_index_flat.ntotal)

k = 4                          # we want to see 4 nearest neighbors
D, I = gpu_index_flat.search(xq, k)  # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries


## Using an IVF index

nlist = 100
quantizer = faiss.IndexFlatL2(d)  # the other index
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
# here we specify METRIC_L2, by default it performs inner-product search

# make it an IVF GPU index
gpu_index_ivf = faiss.index_cpu_to_gpu(res, 0, index_ivf)

assert not gpu_index_ivf.is_trained
gpu_index_ivf.train(xb)        # add vectors to the index
assert gpu_index_ivf.is_trained

gpu_index_ivf.add(xb)          # add vectors to the index
print('---------------------------------')
print(gpu_index_ivf.ntotal)

k = 4                          # we want to see 4 nearest neighbors
D, I = gpu_index_ivf.search(xq, k)  # actual search
print(I[:5])                   # neighbors of the 5 first queries
print('---------------------------------')
print(I[-5:])                  # neighbors of the 5 last queries
print('----------------------------------')
print(len(I))
print(I)