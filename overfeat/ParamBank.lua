require("lfs")
local ffi = require("ffi")
local path = lfs.currentdir()
local libfio = ffi.load(path .. "/overfeat/libParamBank.so")

ffi.cdef[[
int read(float *storage, long offset, long length);
int print(long length);
int init(const char *fpath);
int close(void);
]]

local ParamBank = {}

function ParamBank:read(offset, dims, tensor_in)
   local tensor
   if tensor_in == nil then
      tensor = torch.FloatTensor(torch.LongStorage(dims))
   else
      tensor = tensor_in
   end
   local storage = torch.data(tensor)

   local length = 1
   for i,each in ipairs (dims) do
      length = length * each
   end

   if libfio.read(storage, offset, length) > 0 then
      os.exit()
   end
   return tensor
end

function ParamBank:print(length)
   libfio.print(length)
end

function ParamBank:init(filename)
   if libfio.init(filename) > 0 then
      os.exit()
   end
end

function ParamBank:close()
   if libfio.close() > 0 then
      os.exit()
   end
end

return ParamBank
