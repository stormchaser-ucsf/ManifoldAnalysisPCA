function [prin_angles] = compute_prin_angles_manifold(dataTensor,k)
%function [prin_angles] = compute_prin_angles_manifold(dataTensor,k)
prin_angles=[];
for i=1:size(dataTensor,3)
    Xa = (squeeze(dataTensor(:,:,i)));   
    %[Xa,W] = sphere_data(Xa);     
    [Wai,s1,l1]=pca(Xa,'Centered','off');    
    for j=i+1:size(dataTensor,3)
        Xa = (squeeze(dataTensor(:,:,j)));
        [Waj,s2,l2]=pca(Xa,'Centered','off');    
        [u,s,v]=svd(Wai(:,1:k)'*Waj(:,1:k));
        temp = diag(acos(s))*180/pi;        
        prin_angles = [prin_angles temp];        
    end
end
end


  