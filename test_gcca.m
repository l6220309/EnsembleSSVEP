function results = test_gcca(eeg, model, is_ensemble, fs, list_freqs, pha_val, alpha, beta)
% Test phase of the task-related component analysis (TRCA)-based
% steady-state visual evoked potentials (SSVEPs) detection [1].
%
% function results = test_trca(eeg, model, is_ensemble)
%
% Input:
%   eeg             : Input eeg data 
%                     (# of targets, # of channels, Data length [sample])
%   model           : Learning model for tesing phase of the ensemble 
%                     TRCA-based method
%   is_ensemble     : 0 -> TRCA-based method, 
%                     1 -> Ensemble TRCA-based method (defult: 1)
%
% Output:
%   results         : The target estimated by this method
%
% See also:
%   train_trca.m
%
% Reference:
%   [1] M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao, and T.-P. Jung,
%       "Enhancing detection of SSVEPs for a high-speed brain speller using 
%        task-related component analysis",
%       IEEE Trans. Biomed. Eng, 65(1): 104-112, 2018.
%
% Masaki Nakanishi, 22-Dec-2017
% Swartz Center for Computational Neuroscience, Institute for Neural
% Computation, University of California San Diego
% E-mail: masaki@sccn.ucsd.edu

if ~exist('is_ensemble', 'var') || isempty(is_ensemble)
    is_ensemble = 1; end

if ~exist('model', 'var')
    error('Training model based on TRCA is required. See train_trca().'); 
end

fb_coefs = [1:model.num_fbs].^(-1.25)+0.25;
[num_targs, num_chans, num_smpls, num_block] = size(eeg);
for targ_i = 1:1:model.num_targs
    test_tmp = squeeze(eeg(targ_i, :, :));
    for fb_i = 1:1:model.num_fbs
        % testdata = filterbank(test_tmp, model.fs, fb_i);
        testdata = test_tmp;
        for class_i = 1:1:model.num_targs
            r_tmp = [];
            traindata =  squeeze(model.trains(class_i, fb_i, :, :));
            if ~is_ensemble
                w = squeeze(model.W(fb_i, class_i, :));
                w1 = w(1:num_chans);
                w2 = w(num_chans+1:num_chans+num_chans);
                w3 = w(num_chans+num_chans+1:num_chans+num_chans+10);
            else
                w = squeeze(model.W(fb_i, :, :))';
                w1 = w(1:num_chans,:);
                w2 = w(num_chans+1:num_chans+num_chans,:);
                w3 = w(num_chans+num_chans+1:num_chans+num_chans+10,:);
            end
            Ref=ref_signal_nh(list_freqs(class_i),fs,pha_val(class_i),num_smpls,5);

           %% weighted GCCA
           for w_i = 1:num_targs
                r_tmp1 = corrcoef(testdata'*w1(:,w_i), traindata'*w2(:,w_i));
                r_tmp2 = corrcoef(testdata'*w1(:,w_i), Ref'*w3(:,w_i));
                rho1 = r_tmp1(1,2);
                rho2 = r_tmp2(1,2);
                tmp = sign(rho1).*rho1.^2+sign(rho2).*rho2.^2;
                r_tmp = [r_tmp, tmp];
            end
            w_tmp = 1:num_targs;
            weight = exp(-w_tmp*alpha)+beta;
            r_tmp = sort(r_tmp, 'descend');
            res_tmp = sum(weight.*r_tmp);
            r(fb_i,class_i) = res_tmp;

            % hierarchy GCCA
            [~,~,tmp] = canoncorr(testdata', traindata');
            r_tmp = [r_tmp, tmp(1, 2)];
           for w_i = 1:num_targs
                r_tmp1 = corrcoef(testdata'*w1(:,w_i), traindata'*w2(:,w_i));
                r_tmp2 = corrcoef(testdata'*w1(:,w_i), Ref'*w3(:,w_i));
                rho1 = r_tmp1(1,2);
                rho2 = r_tmp2(1,2);
                tmp = sign(rho1).*rho1.^2+sign(rho2).*rho2.^2;
                r_tmp = [r_tmp, tmp];
            end
            r_tmp = sign(r_tmp).*r_tmp.^2;
            res_tmp = sum(r_tmp);
            r(fb_i,class_i) = res_tmp;

            % compared GCCA
            r_tmp1 = corrcoef(testdata'*w1, traindata'*w2);
            r_tmp2 = corrcoef(testdata'*w1, Ref'*w3);
            rho1 = r_tmp1(1,2);
            rho2 = r_tmp2(1,2);
            r_atmp = sign(rho1).*rho1.^2+sign(rho2).*rho2.^2;
           
           for w_i = 1:num_targs
                r_tmp1 = corrcoef(testdata'*w1(:,w_i), traindata'*w2(:,w_i));
                r_tmp2 = corrcoef(testdata'*w1(:,w_i), Ref'*w3(:,w_i));
                rho1 = r_tmp1(1,2);
                rho2 = r_tmp2(1,2);
                tmp = sign(rho1).*rho1.^2+sign(rho2).*rho2.^2;
                r_tmp(w_i)=tmp;
            end
           
            r1(fb_i,class_i) = max(mean(r_tmp));
            r2(fb_i,class_i) = max(r_atmp);


            % standard GCCA
            r_tmp1 = corrcoef(testdata'*w1, traindata'*w2);
            r_tmp2 = corrcoef(testdata'*w1, Ref'*w3);
            
            rho1 = r_tmp1(1,2);
            rho2 = r_tmp2(1,2);
            
            r_tmp = sign(rho1).*rho1.^2+sign(rho2).*rho2.^2;
            r(fb_i,class_i) = r_tmp;

        end % class_i
    end % fb_i
    
    rho = fb_coefs*r;
    [~, tau] = max(rho);
    results(targ_i) = tau;

	% compared GCCA
   %  rho1 = fb_coefs*r1;
   %  rho2 = fb_coefs*r2;
   % 
   % rho = rho1;
   % for i = 1:size(rho1,2)
   %     if rho1(i) > rho2(i)
   %          rho(i) = rho1(i);
   %     else
   %          rho(i) = rho2(i);
   %     end
   % end
   % 
   %  rho1 = fb_coefs*r1;
   %  rho2 = fb_coefs*r2;
   %  if max(rho1)>max(rho2)
   %      [~, tau] = max((rho1));
   %  else
   %      [~, tau] = max((rho2));
   %  end
   %  results(targ_i) = tau;

end % targ_i