local lbfgsb = {}
local ffi = require 'ffi'

function lbfgsb.init(n_param, m_hist, bounds, lb, ub, print_level)
	lbfgsb.lib = ffi.load("./lbfgsb/liblbfgsb.so")
	ffi.cdef [[
	  void setulb_(int *n, int *m,double *x,double *l,double *u,int *nbd, double *f,
						double *g,double *factr,double *pgtol,double *wa,int *iwa,
						char *task, int *len_task, int *iprint, int *lsave,int *isave,double *dsave);
	  size_t strlen ( const char * str );
	  char * strncpy ( char * destination, const char * source, size_t num );
	  int strncmp ( const char * str1, const char * str2, size_t num );
	]]
	local int_p = ffi.typeof("int[1]")
	local double_p = ffi.typeof("double[1]")
	lbfgsb.len_task, lbfgsb.n, lbfgsb.m, lbfgsb.iprint = int_p(), int_p(), int_p(), int_p()
	lbfgsb.f, lbfgsb.factr, lbfgsb.pgtol = double_p(), double_p(), double_p()
	lbfgsb.task = ffi.new("char[61]")
	lbfgsb.lsave, lbfgsb.nbd, lbfgsb.iwa, lbfgsb.isave = ffi.new("int[4]"), ffi.new("int[" .. n_param .. "]"), ffi.new("int[" .. 3*n_param .. "]"), ffi.new("int[44]")
	lbfgsb.x, lbfgsb.l, lbfgsb.u  = ffi.new("double[" .. n_param .. "]"), ffi.new("double[" .. n_param .. "]"), ffi.new("double[" .. n_param .. "]")
	lbfgsb.g, lbfgsb.dsave = ffi.new("double[" .. n_param .. "]"), ffi.new("double[29]")
	lbfgsb.wa = ffi.new("double[" .. 2*m_hist*n_param + 5*n_param + 11*m_hist*m_hist + 8*m_hist .. "]")

	lbfgsb.n[0] = n_param
	lbfgsb.m[0] = m_hist

	lbfgsb.iprint[0] = print_level;
	lbfgsb.factr[0]=1.0e+7;
    lbfgsb.pgtol[0]=1.0e-5;
    
    for i = 0, n_param - 1 do
		lbfgsb.l[i] = lb[i + 1]
		lbfgsb.u[i] = ub[i + 1]
		lbfgsb.nbd[i] = bounds[i+1]
	end
end

function lbfgsb.eval(feval, w, max_iter)
	local lua_task = "START                                                       "
	ffi.C.strncpy(lbfgsb.task, lua_task, 61);
	lbfgsb.len_task[0] = ffi.C.strlen(lbfgsb.task)

	local count = 0
	for i = 0, lbfgsb.n[0] - 1 do
		lbfgsb.x[i] = w[i + 1]
	end

	repeat
		if lua_task:sub(1,2) == "FG" or lua_task:sub(1,5) == "START" then
			for i= 0, lbfgsb.n[0] - 1 do
				w[i + 1] = lbfgsb.x[i]
			end
			lbfgsb.f[0], lua_g = feval(w)
			count = count + 1
			for i= 0, lbfgsb.n[0] - 1 do
				lbfgsb.g[i] = lua_g[i + 1]
			end
		end
		
		lbfgsb.lib.setulb_(lbfgsb.n,lbfgsb.m,lbfgsb.x,lbfgsb.l,lbfgsb.u,lbfgsb.nbd,lbfgsb.f,lbfgsb.g,lbfgsb.factr,lbfgsb.pgtol,
						   lbfgsb.wa,lbfgsb.iwa,lbfgsb.task,lbfgsb.len_task,lbfgsb.iprint,lbfgsb.lsave,lbfgsb.isave,lbfgsb.dsave);
		lua_task = ffi.string(lbfgsb.task)
	until lua_task:sub(1,2) ~= "FG" and lua_task:sub(1,5) ~= "NEW_X" or count == max_iter
	if count == max_iter then
		print("LBFGSB: too many iterations, reached max of " .. max_iter)
	end
end

return lbfgsb
