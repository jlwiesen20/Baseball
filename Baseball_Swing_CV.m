%Choose googLeNet as pre-trained network
netCNN = googlenet;

%Folder with Input Videos, function at end of script
dataFolder = "Baseball Input Data";
[files,labels] = BaseballFiles(dataFolder);

%Read the video using the readVideo Function at end of script
idx = 1;
filename = files(idx);
video = readVideo(filename);
size(video);

labels(idx);

%Plays a video by sequencing frames of video
numFrames = size(video,4);
figure
for i = 1:numFrames
    frame = video(:,:,:,i);
    imshow(frame/255);
    drawnow
end

%Extract features using GoogLeNet network by getting activations of input data
%Convert videos to sequence of feature vectors
%Feature vectors are output of activation function in last pooling layer of GoogLeNet "pool5-7x7_s1"
inputSize = netCNN.Layers(1).InputSize(1:2);
layerName = "pool5-7x7_s1";

numFiles = numel(files);
sequences = cell(numFiles,1);
    
for i = 1:numFiles
    fprintf("Reading file %d of %d...\n", i, numFiles)
        
    video = readVideo(files(i));
    %GoogLenet needs size 224x224, function at end of script
    video = centerCrop(video,inputSize);
        
    sequences{i,1} = activations(netCNN,video,layerName,'OutputAs','columns');
end
    
    save(tempFile,"sequences","-v7.3");

%Number of array elements "sequences"
numObservations = numel(sequences);
%Random permutation
idx = randperm(numObservations);
%Assign 70% of input data as training data, 30% as validation data
N = floor(0.7 * numObservations);

%Sequences for training
idxTrain = idx(1:N);
sequencesTrain = sequences(idxTrain);
labelsTrain = labels(idxTrain);

%Sequences for validation
idxValidation = idx(N+1:end);
sequencesValidation = sequences(idxValidation);
labelsValidation = labels(idxValidation);

%Get sequence length of training data
numObservationsTrain = numel(sequencesTrain);
sequenceLengths = zeros(1,numObservationsTrain);

for i = 1:numObservationsTrain
    sequence = sequencesTrain{i};
    sequenceLengths(i) = size(sequence,2);
end

%Remove (if any) the training sequences that have more than 400 time steps
maxLength = 400;
idx = sequenceLengths > maxLength;
sequencesTrain(idx) = [];
labelsTrain(idx) = [];

%Number of features per sequence
numFeatures = size(sequencesTrain{1},1);
%2 Classes, Casting Hands and Inside Out
numClasses = numel(categories(labelsTrain));

layers = [
    sequenceInputLayer(numFeatures,'Name','sequence')%Input layer corresponding to number of features per sequence
    bilstmLayer(2000,'OutputMode','last','Name','bilstm')%BiLSTM learns bidirectional long-term dependencies between steps of sequence data.
    dropoutLayer(0.5,'Name','drop')%Dropout layer randomly sets input elements to zero with a given probability.
    fullyConnectedLayer(numClasses,'Name','fc') %Fully connected layer with an output size corresponding to the number of classes
    softmaxLayer('Name','softmax')%Neural transfer function to calculate the layer's output from it's net input
    classificationLayer('Name','classification')];%Compute cross entropy loss %Cross entropy loss measures performance of model

%Mini-batch is a subset of the training set that is used to evaluate the gradient of the loss function and update the weights.
miniBatchSize = 10;
numObservations = numel(sequencesTrain);
%Epoch - Full pass of the training algorithm over the entire training set.
numIterationsPerEpoch = floor(numObservations / miniBatchSize);

%Validate the network once per epoch.
options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ... %
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',.0001, ...
    'GradientThreshold',2, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{sequencesValidation,labelsValidation}, ...
    'ValidationFrequency',numIterationsPerEpoch, ...
    'Plots','training-progress', ...
    'Verbose',false);

%Train the network, observe performance on graph
[netLSTM,info] = trainNetwork(sequencesTrain,labelsTrain,layers,options);

%Calculate the classification accuracy of the network on the validation set.
YPred = classify(netLSTM,sequencesValidation,'MiniBatchSize',miniBatchSize);
YValidation = labelsValidation;
%Logical array of 0 or 1 values. If prediction = validation, then value 1. 
accuracy = mean(YPred == YValidation)

%***Use the layers from the convolutional network to transform the videos
%into vector sequences and the layers from the LSTM network to classify the
%vector sequences.***

%Show graph visualization of network. Use graph to navigate through next steps
cnnLayers = layerGraph(netCNN);

%Remove the data input layer and the layers after the pooling layer used for the activations
layerNames = ["data" "pool5-drop_7x7_s1" "loss3-classifier" "prob" "output"];
cnnLayers = removeLayers(cnnLayers,layerNames);

%Create a sequence input layer that accepts image sequences containing images of the same input size as the GoogLeNet network. 
%To normalize the images using the same average image as the GoogLeNet network, set the 'Normalization' option of the sequence input layer to 'zerocenter' and the 'Mean' option to the average image of the input layer of GoogLeNet.
inputSize = netCNN.Layers(1).InputSize(1:2);
averageImage = netCNN.Layers(1).AverageImage;

inputLayer = sequenceInputLayer([inputSize 3], ...
    'Normalization','zerocenter', ...
    'Mean',averageImage, ...
    'Name','input');

%Add the sequence input layer to the layer graph. 
%Apply convolutional layers to the images of the sequences independently
%Remove the sequence structure of the image sequences by including a sequence folding layer between the sequence input layer and the convolutional layers. 
%Connect the output of the sequence folding layer to the input of the first convolutional layer
layers = [
    inputLayer
    sequenceFoldingLayer('Name','fold')];

lgraph = addLayers(cnnLayers,layers);
lgraph = connectLayers(lgraph,"fold/out","conv1-7x7_s2");

%Add the LSTM layers to the layer graph by removing the sequence input layer of the LSTM network. 
%To restore the sequence structure removed by the sequence folding layer, include a sequence unfolding layer after the convolution layers. 
%The LSTM layers expect sequences of vectors. 
%To reshape the output of the sequence unfolding layer to vector sequences, include a flatten layer after the sequence unfolding layer.
%Take the layers from the LSTM network and remove the sequence input layer.
lstmLayers = netLSTM.Layers;
lstmLayers(1) = [];

%Add the sequence folding layer, the flatten layer, and the LSTM layers to the layer graph. 
%Connect the last convolutional layer ("pool5-7x7_s1") to the input of the sequence unfolding layer ("unfold/in").
layers = [
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    lstmLayers];

lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,"pool5-7x7_s1","unfold/in");

%To enable the unfolding layer to restore the sequence structure, 
%connect the "miniBatchSize" output of the sequence folding layer to the corresponding input of the sequence unfolding layer.
lgraph = connectLayers(lgraph,"fold/miniBatchSize","unfold/miniBatchSize");

%Analyze the network and check for any errors
analyzeNetwork(lgraph)

%Assemble the network
net = assembleNetwork(lgraph);


%Test the network using test videos
filename = "Casting Hands Swing test.mp4";
video = readVideo(filename);

video = centerCrop(video,inputSize);
Class1 = classify(net,{video});

numFrames = size(video,4);
figure(1)

for i = 1:numFrames
    frame = video(:,:,:,i);
    imshow(frame/255);
    title(char(Class1));
    drawnow
end

filename = "Inside Out Swing test.mp4";
video = readVideo(filename);

video = centerCrop(video,inputSize);
Class2 = classify(net,{video});

numFrames = size(video,4);
figure(2)

for i = 1:numFrames
    frame = video(:,:,:,i);
    imshow(frame/255);
    title(char(Class2));
    drawnow
end

filename = "Casting Hands Swing test 2.mp4";
video = readVideo(filename);

video = centerCrop(video,inputSize);
Class3 = classify(net,{video});

numFrames = size(video,4);
figure(3)

for i = 1:numFrames
    frame = video(:,:,:,i);
    imshow(frame/255);
    title(char(Class3));
    drawnow
end

filename = "Inside Out Swing test 2.mp4";
video = readVideo(filename);

video = centerCrop(video,inputSize);
Class4 = classify(net,{video});

numFrames = size(video,4);
figure(4)

for i = 1:numFrames
    frame = video(:,:,:,i);
    imshow(frame/255);
    title(char(Class4));
    drawnow
end

Classifications = [Class1 Class2 Class3 Class4]



%Helper Functions

function [files, labels] = BaseballFiles(dataFolder)
%[files, labels] = BaseballFiles(dataFolder) returns the list of video files and labels (folders) given by dataFolder

fileExtension = ".mp4";
listing = dir(fullfile(dataFolder, "*", "*" + fileExtension));

numObservations = numel(listing);
files = strings(numObservations,1);
labels = cell(numObservations,1);

for i = 1:numObservations
    name = listing(i).name;
    folder = listing(i).folder;
    
    [~,labels{i}] = fileparts(folder);
    files(i) = fullfile(folder,name);
end

labels = categorical(labels);

end

function video = readVideo(filename)

vr = VideoReader(filename);
H = vr.Height;
W = vr.Width;
C = 3;

% Preallocate video array
numFrames = floor(vr.Duration * vr.FrameRate);
video = zeros(H,W,C,numFrames);

% Read frames
i = 0;
while hasFrame(vr)
    i = i + 1;
    video(:,:,:,i) = readFrame(vr);
end

% Remove unallocated frames
if size(video,4) > i
    video(:,:,:,i+1:end) = [];
end

end

function videoResized = centerCrop(video,inputSize)

sz = size(video);

if sz(1) < sz(2)
    % Video is landscape
    idx = floor((sz(2) - sz(1))/2);
    video(:,1:(idx-1),:,:) = [];
    video(:,(sz(1)+1):end,:,:) = [];
    
elseif sz(2) < sz(1)
    % Video is portrait
    idx = floor((sz(1) - sz(2))/2);
    video(1:(idx-1),:,:,:) = [];
    video((sz(2)+1):end,:,:,:) = [];
end

videoResized = imresize(video,inputSize(1:2));

end