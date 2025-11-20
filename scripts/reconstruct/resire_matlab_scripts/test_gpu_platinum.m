parallel.gpu.enableCUDAForwardCompatibility(true)

addpath('functions');
addpath('src');
addpath('data');

%% load data
angles = importdata([pwd,'/data/platinum/platinum_angles.mat']);
projections  = importdata([pwd,'/data/platinum/platinum_projections.mat']);

projections = permute(projections, [2, 3, 1]);
projections = reshape(projections, size(projections));

%% preprocess: rotation setup
rotation       = 'ZYX';  % Euler angles setting ZYX
dtype          = 'single';
projections_refined = cast(projections,dtype);
angles_refined      = cast(angles,dtype);

% compute normal vector of rotation matrix
matR = zeros(3,3);
if length(rotation)~=3
    disp('rotation not recognized. Set rotation = ZYX\n'); rotation = 'ZYX';
end
for i=1:3
    switch rotation(i)
        case 'X',   matR(:,i) = [1;0;0];
        case 'Y',   matR(:,i) = [0;1;0];
        case 'Z',   matR(:,i) = [0;0;1];
        otherwise,  matR = [0,0,1;
                0,1,0;
                1,0,0];
            disp('Rotation not recognized. Set rotation = ZYX');
            break
    end
end
vec1 = matR(:,1); vec2 = matR(:,2); vec3 = matR(:,3);

% extract size of projections & num of projections
disp(size(projections_refined));
[dimx, dimy, Num_pj] = size(projections_refined);

%% rotation matrix
Rs = zeros(3,3,Num_pj, dtype);
for k = 1:Num_pj
    phi   = angles_refined(k,1);
    theta = angles_refined(k,2);
    psi   = angles_refined(k,3);
    
    % compute rotation matrix R w.r.t euler angles {phi,theta,psi}
    rotmat1 = MatrixQuaternionRot(vec1,phi);
    rotmat2 = MatrixQuaternionRot(vec2,theta);
    rotmat3 = MatrixQuaternionRot(vec3,psi);
    R =  single(rotmat1*rotmat2*rotmat3)';
    Rs(:,:,k) = R;
end

%% parameter
step_size      = 2.;  %step_size <=1 but can be larger is sparse
iterations     = 50;
dimz           = dimx;
positivity     = true;

%% iteration: minimize ||Au-b||^2 by gradient descent: run reconstruction the first time
% syntax 1: no initial rec
tic
[rec] = RT3_1GPU( (projections_refined), (Rs), (dimz), ...
    (iterations), (step_size) , (positivity));
toc

%% save result
save('RESIRE_volume_reconstruction.mat', 'rec', '-v7.3');  % v7.3 supports >2 GB
