% (C) Copyright 2011, Liang Xiong (lxiong[at]cs[dot]cmu[dot]edu)
% 
% This piece of software is free for research purposes. 
% We hope it is helpful but do not privide any warranty.
% If you encountered any problems please contact the author.


% complete incomplete/zappos completed/zappos_100_50.mat 100 50 2
% complete incomplete/mitstates completed/mitstates_30_50.mat 30 50 1

function ret = complete(clfs, out, D, n_sample, lim)

	build;
	cd ./lib
	build;
	cd ..
	addpath ./lib

	load(clfs, 'subs', 'vals', 'fullsubs', 'fullvals', 'size')

	% % load TTe
	TTr = spTensor(subs+1, vals, size);
	TTe = spTensor(fullsubs+1, fullvals, size);
	CTr = TTr.Reduce(1:2);
	% CTe = TTe.Reduce(1:2);

	lim = str2num(lim)
	D = str2num(D)
	pn = 50e-3;
	max_iter = 500;
	learn_rate = 1e-4;
	n_sample = str2num(n_sample)

	[U, V, dummy, r_pmf] = PMF_Grad(CTr, [], D, struct('ridge',pn,'learn_rate',learn_rate,'range',[-lim, lim],'max_iter',300));

	alpha = 2;
	[Us_bptf Vs_bptf Ts_bptf] = BPTF(TTr, [], D, struct('Walpha',alpha, 'nuAlpha',1), {U,V,ones(D,TTr.size(3))}, struct('max_iter',max_iter,'n_sample',n_sample,'save_sample',false,'run_name','alpha2-1'));
	[Y_bptf] = BPTF_Predict(Us_bptf,Vs_bptf,Ts_bptf,D,TTe,[-lim, lim]);
	subs = Y_bptf.subs - 1;
	vals = Y_bptf.vals;
	save(out, 'subs', 'vals')


	ret = 0;