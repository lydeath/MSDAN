
clear all;
close all;

gt_path='../datasets/test/Rain100H/';

MSDAN = '../results/Ablation/MSDAN/';
MSDAN6_x = '../results/Ablation/MSDAN6_x/';
MSDAN5_LSTM = '../results/Ablation/MSDAN5_LSTM/';
MSDAN6_LSTM = '../results/Ablation/MSDAN6_LSTM/';
MSDAN7_LSTM = '../results/Ablation/MSDAN7_LSTM/';
MSDAN6_GRU = '../results/Ablation/MSDAN6_GRU/';
MSDAN_RecSSIM = '../results/Ablation/MSDAN_RecSSIM/';

 
struct_model = {
           struct('model_name','MSDAN5_LSTM','path',MSDAN5_LSTM),...
           struct('model_name','MSDAN6_LSTM','path',MSDAN6_LSTM),...
           struct('model_name','MSDAN7_LSTM','path',MSDAN7_LSTM),...
           struct('model_name','MSDAN6_GRU','path',MSDAN6_GRU),...
           struct('model_name','MSDAN','path',MSDAN),...
           struct('model_name','MSDAN_RecSSIM','path',MSDAN_RecSSIM),...
           struct('model_name','MSDAN6_x','path',MSDAN6_x),...
    };


nimgs=100;nrain=1;
nmodel = length(struct_model);

psnrs = zeros(nimgs,nmodel);
ssims = psnrs;

for nnn = 1:nmodel
    
    tp=0;ts=0;te=0;
    nstart = 0;
    for iii=nstart+1:nstart+nimgs
        for jjj=1:nrain
            %         fprintf('img=%d,kernel=%d\n',iii,jjj);
            x_true=im2double(imread(fullfile(gt_path,sprintf('norain-%03d.png',iii))));%x_true
            x_true = rgb2ycbcr(x_true);x_true=x_true(:,:,1);
            

            %%
            x = (im2double(imread(fullfile(struct_model{nnn}.path,sprintf('rain-%03d.png',iii)))));
            x = rgb2ycbcr(x);x = x(:,:,1);
            tp = mean(psnr(x,x_true));
            ts = ssim(x*255,x_true*255);
            
            psnrs(iii-nstart,nnn)=tp;
            ssims(iii-nstart,nnn)=ts;
            
            %
        end
    end
    
    fprintf('%s: psnr=%6.4f, ssim=%6.4f\n',struct_model{nnn}.model_name,mean(psnrs(:,nnn)),mean(ssims(:,nnn)));
    
end

