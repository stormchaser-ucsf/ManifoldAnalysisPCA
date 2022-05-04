%% DEMO for manifold based analyses
clear
clc
close all

% navigate to the folder where the code is
cd('C:\Users\Nikhlesh\Documents\GitHub\ManifoldAnalysisPCA')
addpath(genpath(pwd))


%% ANALYSES 1: PRINCIPLE ANGLES BETWEEN NEURAL MANIFOLDS
% comparing intrinsic manifold between conditions

%%%% simulating 50 channels and 1000 time-points from gaussian:
C = toeplitz([50:-1:1]); % arbritrary covariance structure
Chalf = chol(C); % cholesky sq. root decomposition
X1 = zscore(randn(1000,50)); % mean-centered and scaled gaussian
X1 = X1*Chalf;

% simulate another condition with exactly or noisily similar covariance structure
%X2 = zscore(randn(1000,50))*Chalf; % exactly same cov 
noise_level=0.75; % play around with this...at what point does system fail?
X2 = zscore(randn(1000,50))*(Chalf+triu(noise_level*randn(size(Chalf)))); % noisier version

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
axis tight
xlabel('Dimension')
ylabel('VAF')
subplot(1,3,2)
stem(cumsum(l2)./sum(l2))
title('VAF X2')
axis tight
xlabel('Dimension')
subplot(1,3,3)
stem(cumsum(l3)./sum(l3))
title('VAF X3')
axis tight
xlabel('Dimension')

% look at first PC for each dataset
figure;subplot(1,3,1)
stem(c1(:,1))
axis tight
xlabel('Channel')
ylabel('PC 1 weight')
title('X1-PC1')
subplot(1,3,2)
stem(c2(:,1))
axis tight
xlabel('Channel')
ylabel('PC 1 weight')
title('X2-PC1')
subplot(1,3,3)
stem(c3(:,1))
axis tight
xlabel('Channel')
ylabel('PC 1 weight')
title('X3-PC1')


% compute the principal angles between a 6D manifold based on VAF
dataTensor(:,:,1) = X1;
dataTensor(:,:,2) = X2;
dataTensor(:,:,3) = X3;
dim=6; % dim of manifold
prin_angles = compute_prin_angles_manifold(dataTensor,dim); %column wise, X1 vs X2, X1 vs X3, X2 vs X3

% generate max. entroy statistics of the entire dataset without neural covariance
maxEntropy = run_tme_manifold(dataTensor,'surrogate-TC');

% sample repeatedly and generate surrogate distribution
prin_angles_boot=[];
iter=2000;
for i=1:iter     
    disp([num2str(i/iter*100) '%'])
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
l1 = round(0.025*size(tempb,1)); %2.5th percentile
l2 = round(0.975*size(tempb,1)); % 97.5th percentile
%plot(tempb(1,:),'--b')
plot(tempb(l1,:),'--r')
axis tight
ylim([0 90])
xlim([0.5 size(tempb,2)])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xticklabels ''
yticklabels ''
yticks([0:15:90])
xticks([1:dim])
legend({'Prin angles between X1 and X2','TME lower bound'})
xticklabels(1:dim)
yticklabels(0:15:90)
xlabel('Dimension')
ylabel('Principle angle')

% pvalues per principle angle, testing  if SMALLER than chance
% small pvalues -> reject the null hypothesis of no difference
% interpretation pvalue -> probability of observing a result as extreme given the null hypothesis 
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
    title(['pvalue ' num2str(pval) '    Dimension ' num2str(i)]);
    legend('Null')
end



% plot comparison between 1st and 3rd manifolds with null stats
figure;
hold on
plot(prin_angles(:,2),'k','LineWidth',1)
tempb = prin_angles_boot(:,2,:);
tempb=tempb(:,:);
tempb=sort(tempb');
l1 = round(0.025*size(tempb,1)); %2.5th percentile
l2 = round(0.975*size(tempb,1)); % 97.5th percentile
%plot(tempb(1,:),'--b')
plot(tempb(l1,:),'--r')
axis tight
ylim([0 90])
xlim([0.5 size(tempb,2)])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xticklabels ''
yticklabels ''
yticks([0:15:90])
xticks([1:dim])
legend({'Prin angles between X1 and X3','TME lower bound'})
xticklabels(1:dim)
yticklabels(0:15:90)
xlabel('Dimension')
ylabel('Principle angle')

% pvalues per principle angle, testing if SMALLER than chance
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
    title(['pvalue ' num2str(pval) '    Dimension ' num2str(i)]);
    legend('Null')
end



% plot comparison between 2nd and 3rd manifolds with null stats
figure;
hold on
plot(prin_angles(:,3),'k','LineWidth',1)
tempb = prin_angles_boot(:,3,:);
tempb=tempb(:,:);
tempb=sort(tempb');
l1 = round(0.025*size(tempb,1)); %2.5th percentile
l2 = round(0.975*size(tempb,1)); % 97.5th percentile
%plot(tempb(1,:),'--b')
plot(tempb(l1,:),'--r')
axis tight
ylim([0 90])
xlim([0.5 size(tempb,2)])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xticklabels ''
yticklabels ''
yticks([0:15:90])
xticks([1:dim])
legend({'Prin angles between X2 and X3','TME lower bound'})
xticklabels(1:dim)
yticklabels(0:15:90)
xlabel('Dimension')
ylabel('Principle angle')

% pvalues per principle angle, testing if SMALLER than chance
temp = prin_angles(:,3);
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
    title(['pvalue ' num2str(pval) '    Dimension ' num2str(i)]);
    legend('Null')
end



%% ANALYSES 2: VAF WHEN CROSS PROJECTING DATA FROM ONE CONDN TO MANIFOLD OF ANOTHER
% sticking with the same dummy variables X1, X2, X3 created in prev. cell

dim=6; % still using a 6 dim manifold
neural_vaf=[];
neural_vaf_boot_overall=[];
for i=1:size(dataTensor,3)
    disp(['Processing dataset ' num2str(i)])
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
        
        % principal directions between manifolds
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
% X1 vs X2 -> VAF after cross projection GREATER than chance
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
    xlabel('Neural VAF ratio')
    ylabel('Count')
    legend('Null')
end
disp('')
disp('')
disp('Neural VAF cross-proj ratio b/w: X1 & X2, X1 & X3, X2 & X3')
disp(neural_vaf')


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
disp('')
disp('')
disp('Shared variance between X1 & X2, X1 & X3, X2 & X3')
disp(shared_var_overall')

%% ANALYSIS 4 -> CANONICAL CORRELATION ANALYSIS EXAMPLE (CCA)

clc;clear
close all

% create the two multivariate datasets, time X channels
A = randn(1000,4);
B = randn(1000,2);


% simulate  multilinear, multivariable relationships between the datasets
A(:,2) = A(:,2) + 2*A(:,4);
A(:,1) = randn(1)*A(:,2) + randn(1)*B(:,1) + randn(1)*B(:,2) + A(:,1);
A(:,3) = randn(1)*A(:,4) + randn(1)*B(:,1) + randn(1)*B(:,2) + A(:,3);

% run CCA
[Wa,Wb,S,Za,Zb] = cca(A,B);

% plot the 1st uncovered relationship
figure;
% plot the projected activity on each dataset's CCA manifolds; these are
% maximally correlated
subplot(1,3,1)
plot(Za(:,1),Zb(:,1),'.')
title(['CCA correlation of  ' num2str(S(1))])
xlabel('Proj. on CCA manifold of A')
ylabel('Proj. on CCA manifold of B')

% plot A's CCA manifold...it is a vector in channel space
subplot(1,3,2)
stem(Wa(:,1),'LineWidth',1)
xlabel('Channels')
ylabel('Weight')
title('CCA manifold of A')
xticks(1:size(A,2))
xlim([0.5 size(A,2)+0.5])

% plot B's CCA manifold...it is a vector in channel space
subplot(1,3,3)
stem(Wb(:,1),'LineWidth',1)
title('CCA manifold of B')
ylabel('Weight')
xlabel('Channels')
xticks(1:size(B,2))
xlim([0.5 size(B,2)+0.5])



