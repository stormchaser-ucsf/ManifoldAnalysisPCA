%% DEMO for manifold based analyses
clear
clc
close all

% navigate to the folder where the code is
cd('C:\Users\Nikhlesh\Documents\GitHub\ManifoldAnalyses')
addpath(genpath(pwd))


%% ANALYSES 1: PRINCIPLE ANGLES BETWEEN NEURAL MANIFOLDS
% comparing intrinsic manifold between conditions

%%%% simulating 50D gaussian data with 1000 time-points:
C = toeplitz([50:-1:1]); % arbritrary covariance structure
Chalf = chol(C); % cholesky sq. root decomposition
X1 = zscore(randn(1000,50)); % mean-centered and scaled gaussian
X1 = X1*Chalf;

% simulate another condition with similar covariance structure
X2 = zscore(randn(1000,50))*Chalf;

% simulate another conditon wit no covariance structure but same variances
X3 = zscore(randn(1000,50))*chol(50*eye(50));


% visualize
figure;
subplot(2,2,1)
imagesc(C)
colorbar
title('Cov structure to be simulated')
subplot(2,2,2)
imagesc(cov(X1))
title('Simulated Cov structure X1')
colorbar
subplot(2,2,3)
imagesc(cov(X2))
title('Simulated Cov structure X2')
colorbar
subplot(2,2,4)
imagesc(cov(X3))
title('Simulated Cov structure X3')
colorbar

% define manifold by principal component analyses
[c1,s1,l1]=pca(X1);
[c2,s2,l2]=pca(X2);
[c3,s3,l3]=pca(X3);

% visualize variance accounted for for each dataset
figure;
subplot(1,3,1)
stem(cumsum(l1)./sum(l1))
title('VAF X1')
subplot(1,3,2)
stem(cumsum(l2)./sum(l2))
title('VAF X2')
subplot(1,3,3)
stem(cumsum(l3)./sum(l3))
title('VAF X3')

% compute the principal angles between a 6D manifold based on VAF
dataTensor(:,:,1) = X1;
dataTensor(:,:,2) = X2;
dataTensor(:,:,3) = X3;
dim=6; % dim of manifold
prin_angles = compute_prin_angles_manifold(dataTensor,dim);

% generate max. entroy statistics of the entire dataset without neural covariance
maxEntropy = run_tme_manifold(dataTensor,'surrogate-TC');

% sample repeatedly and generate null distribution
prin_angles_boot=[];
for i=1:2000
    disp(i)
    surrTensor = sampleTME(maxEntropy);
    prin_angles_boot = cat(3,compute_prin_angles_manifold(surrTensor,dim),...
        prin_angles_boot);
end

% plot comparison between 1st and 2nd manifolds with null stats
figure;
hold on
plot(prin_angles(:,1),'k','LineWidth',1)
tempb = prin_angles_boot(:,1,:);
tempb=tempb(:,:);
tempb=sort(tempb');
%l1 = round(0.025*size(tempb,1)); %2.5th percentile
%l2 = round(0.975*size(tempb,1)); % 97.5th percentile
plot(tempb(1,:),'--b')
plot(tempb(end,:),'--r')
axis tight
ylim([0 90])
xlim([0.5 size(tempb,2)])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xticklabels ''
yticklabels ''
yticks([0:15:90])
xticks([1:5])
legend({'Prin angles between X1 and X2','Low null','High null'})

% pvalues per principle angle, testing if smaller than chance
temp = prin_angles(:,1);
for i=1:length(temp)
    figure;hist(tempb(:,i));
    vline(temp(i),'r')
    xlim([0 90])
    xlabel('Angle')
    ylabel('Count')
    pval = sum(temp(i)>tempb(:,i))/length(tempb(:,i));
    if pval==0
        pval=1/length(tempb(:,i));
    end
    title(['pvalue ' num2str(pval)]);
end



% plot comparison between 1st and 3rd manifolds with null stats
figure;
hold on
plot(prin_angles(:,2),'k','LineWidth',1)
tempb = prin_angles_boot(:,2,:);
tempb=tempb(:,:);
tempb=sort(tempb');
%l1 = round(0.025*size(tempb,1)); %2.5th percentile
%l2 = round(0.975*size(tempb,1)); % 97.5th percentile
plot(tempb(1,:),'--b')
plot(tempb(end,:),'--r')
axis tight
ylim([0 90])
xlim([0.5 size(tempb,2)])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xticklabels ''
yticklabels ''
yticks([0:15:90])
xticks([1:5])
legend({'Prin angles between X1 and X3','Low null','High null'})

% pvalues per principle angle, testing if smaller than chance
temp = prin_angles(:,2);
for i=1:length(temp)
    figure;hist(tempb(:,i));
    vline(temp(i),'r')
    xlim([0 90])
    xlabel('Angle')
    ylabel('Count')
    pval = sum(temp(i)>tempb(:,i))/length(tempb(:,i));
    if pval==0
        pval=1/length(tempb(:,i));
    end
    title(['pvalue ' num2str(pval)]);
end



%% ANALYSES 2: VAF WHEN CROSS PROJECTING DATA FROM ONE CONDN TO MANIFOLD OF ANOTHER
% sticking with the same dummy variables X1, X2, X3 created in prev. cell

dim=6; % still using a 6 dim manifold
neural_vaf=[];
neural_vaf_boot_overall=[];
for i=1:size(dataTensor,3)
    Xa = squeeze(dataTensor(:,:,i));
    
    % compute manifold via PCA and VAF for first 'dim' modes
    [c1,s1,l1] = pca(Xa,'Centered','off');
    l1 =  cumsum(l1)/sum(l1);
    vaf_xa = l1(dim);
    
    for j=i+1:size(dataTensor,3)
        vaf_ratio = [];
        
        Xb = squeeze(dataTensor(:,:,j));
        
        % compute manifold via PCA and VAF for first 'dim' modes
        [c2,s2,l2] = pca(Xb,'Centered','off');
        l2 =  cumsum(l2)/sum(l2);
        vaf_xb = l2(dim);
        
        % principal angles between manifolds
        [Pa,S,Pb ] = svd(c1(:,1:dim)'*c2(:,1:dim));
        
        % overall variance
        xa = sum(Xa(:).^2);
        xb = sum(Xb(:).^2);
        
        % projecting i on j
        enc = c2(:,1:dim)*Pb;
        dec = enc';
        recon = Xa - Xa*enc*dec;
        recon_vaf = sum(recon(:).^2);
        recon_vaf = (xa - recon_vaf)/xa;
        vaf_ratio = [vaf_ratio;recon_vaf/vaf_xa];
        
        % projecting j on i
        enc = c1(:,1:dim)*Pa;
        dec = enc';
        recon = Xb - Xb*enc*dec;
        recon_vaf = sum(recon(:).^2);
        recon_vaf = (xb - recon_vaf)/xb;
        vaf_ratio = [vaf_ratio;recon_vaf/vaf_xb];
        
        % storing results
        neural_vaf = [neural_vaf; mean(vaf_ratio)];
        
        % projecting onto random manifolds and get distrib. of max. possible VAF
        neural_vaf_boot=[];
        for loop=1:1000
            vaf_ratio_boot=[];
            for iter=1:20
                
                % creating artificial orthonormal manifolds and cross-projecting
                enc1 = randn(size(enc));
                [enc1,~]=qr(enc1,0);
                dec1 = pinv(enc1);
                recon = Xa - Xa*enc1*dec1;
                recon_vaf = sum(recon(:).^2);
                recon_vaf = (xa - recon_vaf)/xa;
                vaf_ratio_boot = [vaf_ratio_boot;recon_vaf/vaf_xa];
                
                % creating artificial orthonormal manifolds and cross-projecting
                enc1 = randn(size(enc));
                [enc1,~]=qr(enc1,0);
                dec1 = pinv(enc1);
                recon = Xb - Xb*enc1*dec1;
                recon_vaf = sum(recon(:).^2);
                recon_vaf = (xb - recon_vaf)/xb;
                vaf_ratio_boot = [vaf_ratio_boot;recon_vaf/vaf_xb];
                
            end
            neural_vaf_boot = [neural_vaf_boot;max(vaf_ratio_boot)]; % can also take median() here
        end
        neural_vaf_boot_overall = cat(2,neural_vaf_boot_overall,neural_vaf_boot);
    end
end


% plotting the statistics with p-values, in following order:
% X1 vs X2 -> VAF after cross projection greater than chance
% X1 vs X3 -> not greater
% X2 vs X3 -> not greater
for i=1:length(neural_vaf)
    figure;hist( neural_vaf_boot_overall(:,i));
    vline(neural_vaf(i),'r');
    xlim([0 1.1])
    pval = 1-sum(neural_vaf(i)>neural_vaf_boot_overall(:,i))/length(neural_vaf_boot_overall(:,i));
    if pval==0
        pval=1/length(neural_vaf_boot_overall(:,i));
    end
    title(['pvalue of ' num2str(pval)])
end


%% ANALYSIS 3 -> SHARED VARIANCE BETWEEN ANY TWO MANIFOLDS
% sticking with the same dummy variables X1, X2, X3 created in prev. cell

dim=6;
shared_var_overall=[];
for i=1:size(dataTensor,3)
    
    Xa = squeeze(dataTensor(:,:,i));
    % compute manifold via PCA
    [c1,s1,l1] = pca(Xa,'Centered','off');
    
    
    for j=i+1:size(dataTensor,3)
        shared_var=[];
        
        Xb = squeeze(dataTensor(:,:,j));
        % compute manifold via PCA
        [c2,s2,l2] = pca(Xb,'Centered','off');
        
        
        % getting shared variance
        % projection 1
        [Q]= c1(:,1:dim);
        [Qm]= c2(:,1:dim);
        num = trace(Q*Q' * Qm*Qm' * Q*Q');
        den = trace(Qm*Qm');
        shared_var = [shared_var num/den];
        
        % projection 2
        [Q]= c2(:,1:dim);
        [Qm]= c1(:,1:dim);
        num = trace(Q*Q' * Qm*Qm' * Q*Q');
        den = trace(Qm*Qm');
        shared_var = [shared_var num/den];
        
        shared_var_overall = [shared_var_overall; mean(shared_var)];
    end
end

disp('Shared variance between X1 & X2, X1 & X3, X2 & X3')
disp(shared_var_overall')

