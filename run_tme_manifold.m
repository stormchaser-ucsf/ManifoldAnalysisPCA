function maxEntropy = run_tme_manifold(dataTensor,surr_type)
%function maxEntropy = run_tme(dataTensor,surr_type)
surrogate_type = surr_type;
[targetSigmaT, targetSigmaN, targetSigmaC, M] = extractFeatures(dataTensor);
numSurrogates = 100;
params = [];
% params.readout_mode = 2;     % select readout mode (eg channel mode)
% params.shfl_mode = 3;        % shuffle across tensor mode (eg condition mode)
% params.fix_mode = 2;         % shuffle per mode (shuffle for each channel independently)
if strcmp(surrogate_type, 'surrogate-T')
    params.margCov{1} = targetSigmaT;
    params.margCov{2} = [];
    params.margCov{3} = [];
    params.meanTensor = M.T;
elseif strcmp(surrogate_type, 'surrogate-TN')
    params.margCov{1} = targetSigmaT;
    params.margCov{2} = targetSigmaN;
    params.margCov{3} = [];
    params.meanTensor = M.TN;
elseif strcmp(surrogate_type, 'surrogate-TNC')
    params.margCov{1} = targetSigmaT;
    params.margCov{2} = targetSigmaN;
    params.margCov{3} = targetSigmaC;
    params.meanTensor = M.TNC;
elseif strcmp(surrogate_type, 'surrogate-NC')
    params.margCov{1} = [];
    params.margCov{2} = targetSigmaN;
    params.margCov{3} = targetSigmaC;
    params.meanTensor = M.NC;
elseif strcmp(surrogate_type, 'surrogate-TC')
    params.margCov{1} = targetSigmaT;
    params.margCov{2} = [];
    params.margCov{3} = targetSigmaC;
    params.meanTensor = M.TC;
else
    error('please specify a correct surrogate type')
end
maxEntropy = fitMaxEntropy(params);
end