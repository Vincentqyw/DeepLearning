
% This is simple demo of restructing Resnet34 using
% Matlab Deeplearning toolbox
% ENV: Matlab 2017a AND above!

%% begin to define each layer
layers = [ 
    %% data - pool1
    imageInputLayer([224 224 3],'Name','data')
    convolution2dLayer(7,64,'Stride', 2,'Padding', 3,'Name','conv1')
    batchNormalizationLayer('Name','bn_conv1')
    reluLayer('Name','conv1_relu')
    
    %% pool1 - res2a
    maxPooling2dLayer(3, 'Stride', 2,'Name','pool1');
    
%     %  left   
%     convolution2dLayer(1,64,'Stride', 1,'Padding', 0,'Name','res2a_branch1')
%     batchNormalizationLayer('Name','bn2a_branch1')
    
    %  right 1
    
    convolution2dLayer(3,64,'Stride', 1,'Padding', 1,'Name','res2a_branch2a')
    batchNormalizationLayer('Name','bn2a_branch2a')
    
    reluLayer('Name','res2a_branch2a_relu')
    
    %  right 2
    convolution2dLayer(3,64,'Stride', 1,'Padding', 1,'Name','res2a_branch2b')
    batchNormalizationLayer('Name','bn2a_branch2b')
    
    
    % add together
    additionLayer(2,'Name','res2a')
    reluLayer('Name','res2a_relu')
    
    
    % res2a - res2b
    % right 1
    convolution2dLayer(3,64,'Stride', 1,'Padding', 1,'Name','res2b_branch2a')
    batchNormalizationLayer('Name','bn2b_branch2a')
    
    reluLayer('Name','res2b_branch2a_relu')
    
    % right 2
    convolution2dLayer(3,64,'Stride', 1,'Padding', 1,'Name','res2b_branch2b')
    batchNormalizationLayer('Name','bn2b_branch2b')
    
    additionLayer(2,'Name','res2b')
    reluLayer('Name','res2b_relu')
    
    % res2b - res2c
    
    % right 1
    convolution2dLayer(3,64,'Stride', 1,'Padding', 1,'Name','res2c_branch2a')
    batchNormalizationLayer('Name','bn2c_branch2a')
    
    reluLayer('Name','res2c_branch2a_relu')
    
    % right 2
    convolution2dLayer(3,64,'Stride', 1,'Padding', 1,'Name','res2c_branch2b')
    batchNormalizationLayer('Name','bn2c_branch2b')
    
    additionLayer(2,'Name','res2c')
    reluLayer('Name','res2c_relu')
    
    % res2c - res3a    
%     % left
%     convolution2dLayer(1,128,'Stride', 2,'Padding', 0,'Name','res3a_branch1')
%     batchNormalizationLayer('Name','bn3a_branch1')
    
    % right 1
    
    convolution2dLayer(3,128,'Stride', 2,'Padding', 1,'Name','res3a_branch2a')
    batchNormalizationLayer('Name','bn3a_branch2a')
    
    reluLayer('Name','res3a_branch2a_relu')
    
    % right 2
    convolution2dLayer(3,128,'Stride', 1,'Padding', 1,'Name','res3a_branch2b')
    batchNormalizationLayer('Name','bn3a_branch2b')
    
    
    additionLayer(2,'Name','res3a')
    reluLayer('Name','res3a_relu')
    
    %% res3a - res3b
    
    % right 1
    convolution2dLayer(3,128,'Stride', 1,'Padding', 1,'Name','res3b_branch2a')
    batchNormalizationLayer('Name','bn3b_branch2a')
    
    reluLayer('Name','res3b_branch2a_relu')
    
    % right 2
    convolution2dLayer(3,128,'Stride', 1,'Padding', 1,'Name','res3b_branch2b')
    batchNormalizationLayer('Name','bn3b_branch2b')
 
    additionLayer(2,'Name','res3b')
    reluLayer('Name','res3b_relu')
    
    %% res3b - res3c
    
    % right 1
    convolution2dLayer(3,128,'Stride', 1,'Padding', 1,'Name','res3c_branch2a')
    batchNormalizationLayer('Name','bn3c_branch2a')
    
    reluLayer('Name','res3c_branch2a_relu')
    
    % right 2
    convolution2dLayer(3,128,'Stride', 1,'Padding', 1,'Name','res3c_branch2b')
    batchNormalizationLayer('Name','bn3c_branch2b')
    
    additionLayer(2,'Name','res3c')
    reluLayer('Name','res3c_relu')
    
    %% res3c - res3d
    
    % right 1
    convolution2dLayer(3,128,'Stride', 1,'Padding', 1,'Name','res3d_branch2a')
    batchNormalizationLayer('Name','bn3d_branch2a')
    
    reluLayer('Name','res3d_branch2a_relu')
    
    % right 2
    convolution2dLayer(3,128,'Stride', 1,'Padding', 1,'Name','res3d_branch2b')
    batchNormalizationLayer('Name','bn3d_branch2b')
    
    additionLayer(2,'Name','res3d')
    reluLayer('Name','res3d_relu')
    
    
    %% res3d - res4a
    
%     % left 
%     convolution2dLayer(1,256,'Stride', 2,'Padding', 0,'Name','res4a_branch1')
%     batchNormalizationLayer('Name','bn4a_branch1')
    
    % right 1
    convolution2dLayer(3,256,'Stride', 2,'Padding', 1,'Name','res4a_branch2a')
    batchNormalizationLayer('Name','bn4a_branch2a')
    
    reluLayer('Name','res4a_branch2a_relu')
    
    % right 2
    convolution2dLayer(3,256,'Stride', 1,'Padding', 1,'Name','res4a_branch2b')
    batchNormalizationLayer('Name','bn4a_branch2b')
    
    additionLayer(2,'Name','res4a')
    reluLayer('Name','res4a_relu')
    
    %% res4a - res4b
    
    % right 1
    convolution2dLayer(3,256,'Stride', 1,'Padding', 1,'Name','res4b_branch2a')
    batchNormalizationLayer('Name','bn4b_branch2a')
    
    reluLayer('Name','res4b_branch2a_relu')
    
    % right 2
    convolution2dLayer(3,256,'Stride', 1,'Padding', 1,'Name','res4b_branch2b')
    batchNormalizationLayer('Name','bn4b_branch2b')
 
    additionLayer(2,'Name','res4b')
    reluLayer('Name','res4b_relu')
    
    %% res4b - res4c
    
    % right 1
    convolution2dLayer(3,256,'Stride', 1,'Padding', 1,'Name','res4c_branch2a')
    batchNormalizationLayer('Name','bn4c_branch2a')
    
    reluLayer('Name','res4c_branch2a_relu')
    
    % right 2
    convolution2dLayer(3,256,'Stride', 1,'Padding', 1,'Name','res4c_branch2b')
    batchNormalizationLayer('Name','bn4c_branch2b')
    
    additionLayer(2,'Name','res4c')
    reluLayer('Name','res4c_relu')
    
    %% res4c - res4d
    
    % right 1
    convolution2dLayer(3,256,'Stride', 1,'Padding', 1,'Name','res4d_branch2a')
    batchNormalizationLayer('Name','bn4d_branch2a')
    
    reluLayer('Name','res4d_branch2a_relu')
    
    % right 2
    convolution2dLayer(3,256,'Stride', 1,'Padding', 1,'Name','res4d_branch2b')
    batchNormalizationLayer('Name','bn4d_branch2b')
    
    additionLayer(2,'Name','res4d')
    reluLayer('Name','res4d_relu')

        
    %% res4d - res4e
    
    % right 1
    convolution2dLayer(3,256,'Stride', 1,'Padding', 1,'Name','res4e_branch2a')
    batchNormalizationLayer('Name','bn4e_branch2a')
    
    reluLayer('Name','res4e_branch2a_relu')
    
    % right 2
    convolution2dLayer(3,256,'Stride', 1,'Padding', 1,'Name','res4e_branch2b')
    batchNormalizationLayer('Name','bn4e_branch2b')
    
    additionLayer(2,'Name','res4e')
    reluLayer('Name','res4e_relu')
    
        
    %% res4e - res4f
    
    % right 1
    convolution2dLayer(3,256,'Stride', 1,'Padding', 1,'Name','res4f_branch2a')
    batchNormalizationLayer('Name','bn4f_branch2a')
    
    reluLayer('Name','res4f_branch2a_relu')
    
    % right 2
    convolution2dLayer(3,256,'Stride', 1,'Padding', 1,'Name','res4f_branch2b')
    batchNormalizationLayer('Name','bn4f_branch2b')
    
    additionLayer(2,'Name','res4f')
    reluLayer('Name','res4f_relu')
    
         
    %% res4f - res5a
    
%     % left 1
%     convolution2dLayer(1,512,'Stride', 2,'Padding', 0,'Name','res5a_branch1')
%     batchNormalizationLayer('Name','bn5a_branch1')
    
    % right 1
    convolution2dLayer(3,512,'Stride', 2,'Padding', 1,'Name','res5a_branch2a')
    batchNormalizationLayer('Name','bn5a_branch2a')
    
    reluLayer('Name','res5a_branch2a_relu')
    
    % right 2
    convolution2dLayer(3,512,'Stride', 1,'Padding', 1,'Name','res5a_branch2b')
    batchNormalizationLayer('Name','bn5a_branch2b')
    
    additionLayer(2,'Name','res5a')
    reluLayer('Name','res5a_relu')
       
    
    %% res5a - res5b
    
    % right 1
    convolution2dLayer(3,512,'Stride', 1,'Padding', 1,'Name','res5b_branch2a')
    batchNormalizationLayer('Name','bn5b_branch2a')
    
    reluLayer('Name','res5b_branch2a_relu')
    
    % right 2
    convolution2dLayer(3,512,'Stride', 1,'Padding', 1,'Name','res5b_branch2b')
    batchNormalizationLayer('Name','bn5b_branch2b')
 
    additionLayer(2,'Name','res5b')
    reluLayer('Name','res5b_relu')
    
    %% res5b - res5c
    
    % right 1
    convolution2dLayer(3,512,'Stride', 1,'Padding', 1,'Name','res5c_branch2a')
    batchNormalizationLayer('Name','bn5c_branch2a')
    
    reluLayer('Name','res5c_branch2a_relu')
    
    % right 2
    convolution2dLayer(3,512,'Stride', 1,'Padding', 1,'Name','res5c_branch2b')
    batchNormalizationLayer('Name','bn5c_branch2b')
    
    additionLayer(2,'Name','res5c')
    reluLayer('Name','res5c_relu')
    
    
    averagePooling2dLayer(1, 'Stride',1,'Name','pool5'); %7
    
    fullyConnectedLayer(10,'Name','fc10');
    
    softmaxLayer('Name','loss');
    classificationLayer('Name','out');
    
];

%% show DAG without shortcut

lgraph = layerGraph(layers);
figure
plot(lgraph)


%% add some connections (shortcut)
layers_2a=[
    convolution2dLayer(1,64,'Stride', 1,'Padding', 0,'Name','res2a_branch1')
    batchNormalizationLayer('Name','bn2a_branch1')
];

lgraph = addLayers(lgraph,layers_2a);
lgraph = connectLayers(lgraph,'pool1','res2a_branch1');    
lgraph = connectLayers(lgraph,'bn2a_branch1','res2a/in2');


lgraph = connectLayers(lgraph,'res2a_relu','res2b/in2');
lgraph = connectLayers(lgraph,'res2b_relu','res2c/in2');


layers_3a=[
    convolution2dLayer(1,128,'Stride', 2,'Padding', 0,'Name','res3a_branch1')
    batchNormalizationLayer('Name','bn3a_branch1')
];

lgraph = addLayers(lgraph,layers_3a);
lgraph = connectLayers(lgraph,'res2c_relu','res3a_branch1');    
lgraph = connectLayers(lgraph,'bn3a_branch1','res3a/in2');

lgraph = connectLayers(lgraph,'res3a_relu','res3b/in2');
lgraph = connectLayers(lgraph,'res3b_relu','res3c/in2');
lgraph = connectLayers(lgraph,'res3c_relu','res3d/in2');


layers_4a=[
    convolution2dLayer(1,256,'Stride', 2,'Padding', 0,'Name','res4a_branch1')
    batchNormalizationLayer('Name','bn4a_branch1')
];

lgraph = addLayers(lgraph,layers_4a);
lgraph = connectLayers(lgraph,'res3d_relu','res4a_branch1');    
lgraph = connectLayers(lgraph,'bn4a_branch1','res4a/in2');

lgraph = connectLayers(lgraph,'res4a_relu','res4b/in2');
lgraph = connectLayers(lgraph,'res4b_relu','res4c/in2');
lgraph = connectLayers(lgraph,'res4c_relu','res4d/in2');
lgraph = connectLayers(lgraph,'res4d_relu','res4e/in2');
lgraph = connectLayers(lgraph,'res4e_relu','res4f/in2');


layers_5a=[
    convolution2dLayer(1,512,'Stride', 2,'Padding', 0,'Name','res5a_branch1')
    batchNormalizationLayer('Name','bn5a_branch1')
];

lgraph = addLayers(lgraph,layers_5a);
lgraph = connectLayers(lgraph,'res4f_relu','res5a_branch1');    
lgraph = connectLayers(lgraph,'bn5a_branch1','res5a/in2');

lgraph = connectLayers(lgraph,'res5a_relu','res5b/in2');
lgraph = connectLayers(lgraph,'res5b_relu','res5c/in2');

% lgraph = connectLayers(lgraph,'data','out');

resnet34_model=lgraph;

%% show net
figure
set(gcf,'color',[1 1 1])
plot(resnet34_model)

