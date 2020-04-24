import numpy as np


a = np.zeros((3, 4, 2, 1))
a[1, 2, 0, 0] = 1
a[1, 2, 1, 0] = 2
a[1, 1, 1, 0] = 3
a[0, 1, 1, 0] = 4

# print(a)
# mask = (a == np.amax(a, axis=(0, 1,)))
# a = a*mask
# print(a)
# print(mask)


# print(a)

# # print(np.unravel_index(np.argmax(a[n,...]), a[n,...].shape))
b = a.reshape((a.shape[0]*a.shape[1], a.shape[2], a.shape[3]))
# print(b)
indx = np.argmax(b, axis=0)

# print(indx)
c = np.array(np.unravel_index(indx, (a.shape[0], a.shape[1]))).astype(int)

print("###")
print(c)

# # d = np.zeros((1, 1, 5, 2))
# # c = np.expand_dims(c, 0)
# print(c[0, :, :], c[1, :, :])
# # print("####")
# # print(c)
# # print(c.shape)

# # z = np.array(np.split(c, 0, axis=0))
# # z = c.reshape((1, 1, c.shape[2], c.shape[3]))
# # z = np.stack(c[0, :, :], c[1, :, :])
# # print(z)
# # print(z.shape)
# print("---------")
# z = np.zeros(a.shape)
# z[c] = np.ones((5, 2))
# print(z.shape)
# print(a)
# print(z)
# # print(a[c[0, :, :], c[1, :, :], :, :].shape)
# # print(a.shape)

# # f = np.einsum("jk,jk->iljk", c[0, ...], c[1, ...])

# # print(f.shape)

# # z = np.zeros((3, 4, 5, 2))[c[0, ...], c[1,]]

# # print(c[0, 2, 1])
# # print(c[1, 2, 1])

# # max_idx = a.reshape((a.shape[0]*a.shape[1], a.shape[2], a.shape[3])).argmax(1)
# # print(max_idx)
# # print(np.unravel_index(max_idx, a[0, 0,:,:].shape))
# # maxpos_vect = np.column_stack(np.unravel_index(max_idx, a[0, 0,:,:].shape))
# # print(maxpos_vect)
