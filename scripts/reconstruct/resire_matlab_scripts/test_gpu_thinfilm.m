parallel.gpu.enableCUDAForwardCompatibility(true)

addpath('functions');
addpath('src');
addpath('data');

%% load data
projections  = importdata([pwd '/data/thinfilm/proj_crop3.mat' ]);
angles       = importdata([pwd '/data/thinfilm/final_ang.mat' ]);
support      = importdata([pwd '/data/thinfilm/support_s3.mat']);

%% rotation setting
rotation       = 'ZYX';  % Euler angles setting ZYZ
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
step_size      = 1.;  %step_size <=1 but can be larger is sparse
iterations     = 200;
dimz           = dimx;
positivity     = true;

%% Test resire2 code on thinfilm with multi-GPU
fprintf('\nResire code:thinfilm (multi-GPU)\n');
dim = [dimx, dimy, dimz];
tic
[rec4] = RT3_1GPU( (projections_refined), (Rs), dim, ...
    (iterations), (step_size/2) , (positivity) );

toc
disp(size(rec4))
rec_yz = permute(rec4, [3,2,1]);
save('rec_thinfilm.mat', 'rec_yz');

