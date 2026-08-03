ldr x8, [x1]
mov w9, #-1
crc32cx w8, w9, x8
ldp w10, w9, [x22, #4]
lsr w8, w8, w10
and w2, w8, w9
