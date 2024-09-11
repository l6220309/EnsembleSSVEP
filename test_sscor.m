function results = test_sscor(eeg, model, is_ensemble, alpha, beta)
% Test phase of the sum of squared correlations (SSCOR)-based
% steady-state visual evoked potentials (SSVEPs) detection [1].
% * This script is the exact same as test_trca.m except for the function
% name.
%
% function results = test_sscor(eeg, model, is_ensemble)
%
% Input:
%   eeg             : Input eeg data 
%                     (# of targets, # of channels, Data length [sample])
%   model           : Learning model for tesing phase of the ensemble 
%                     SSCOR-based method
%   is_ensemble     : 0 -> SSCOR-based method, 
%                     1 -> Ensemble SSCOR-based method (defult: 1)
%
% Output:
%   results         : The target estimated by this method
%
% See also:
%   train_sscor.m
%
% Reference:
%   [1] G. R. Kumar and M. R. Reddy,
%       "Designing a Sum of Squared Correlations Framework for Enhancing SSVEP
%        Based BCIs",
%       IEEE Trans. Neural Syst. Rehabil. Eng., vol. 27, pp. 2044-2050, 2019.
%
% Kuan-Jung Chiang and Masaki Nakanishi, 25-Nov-2019
% Swartz Center for Computational Neuroscience, Institute for Neural
% Computation, University of California San Diego
% E-mail: masaki@sccn.ucsd.edu

if ~exist('is_ensemble', 'var') || isempty(is_ensemble)
    is_ensemble = 1; end

if ~exist('model', 'var')
    error('Training model based on SSCOR is required. See train_sscor().'); 
end

fb_coefs = [1:model.num_fbs].^(-1.25)+0.25;

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
            else
                w = squeeze(model.W(fb_i, :, :))';
            end

            % standard SSCOR
            r_tmp = corrcoef(testdata'*w, traindata'*w);
            r(fb_i,class_i) = r_tmp(1,2);
            
            %% weighted SSCOR
           for w_i = 1:size(w, 2)
                tmp = corrcoef(testdata'*w(:,w_i), traindata'*w(:,w_i));
                r_tmp = [r_tmp, tmp(1,2)];
            end
            w_tmp = 1:size(w, 2);
            weight = exp(-w_tmp*alpha)+beta;
            r_tmp = sort(r_tmp, 'descend');
            res_tmp = sum(weight.*r_tmp);
            r(fb_i,class_i) = res_tmp;

            % hierarchy SSCOR
            [~,~,tmp] = canoncorr(testdata', traindata');
            r_tmp = [r_tmp, tmp(1, 2)];
            for w_i = 1:size(w, 2)
                tmp = corrcoef(testdata'*w(:,w_i), traindata'*w(:,w_i));
                r_tmp = [r_tmp, tmp(1, 2)];
            end
            r_tmp = sign(r_tmp).*r_tmp.^2;
            res_tmp = sum(r_tmp);
            r(fb_i,class_i) = res_tmp;
            
            % compared SSCOR
            test=testdata'*w; ref=traindata'*w;
            r_tmp1 = corrcoef(test, ref);
            for l= 1:size(test, 2)
                r_tm= corrcoef(test(:,l), ref(:,l));
                r_tmp(l)=r_tm(1,2);
            end
            r1(fb_i,class_i) = max(mean(r_tmp));
            r2(fb_i,class_i) = max(r_tmp1(1,2));
            
        end % class_i
    end % fb_i
    
    
    rho = fb_coefs*r;
    [~, tau] = max(rho);
    results(targ_i) = tau;

	% compared SSCOR
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