load Iris-Targets.mat
load Iris-Inputs.mat

x = IrisInput;
t = IrisTargets;

nn = 10; 
n = 0.01;
i = 1;
%trainlm trainbr trainbfg trainrp trainscg traincgb 
%traincgf traincgp trainoss traingdx traingdm traingd

%Função de treinamento
tab_lm = ["trainlm" "mse"; "trainlm" "sse"];
tab_br = ["trainbr" "mse"; "trainbr" "sse"];
tab_gd = ["traingd" "mse"; "traingd" "mae"; "traingd" "sse"; "traingd" "sae"; "traingd" "crossentropy"];
tab_rp = ["trainrp" "mse"; "trainrp" "mae"; "trainrp" "sse"; "trainrp" "sae"; "trainrp" "crossentropy"];
tab_scg = ["trainscg" "mse"; "trainscg" "mae"; "trainscg" "sse"; "trainscg" "sae"; "trainscg" "crossentropy"];
tab_cgb = ["traincgb" "mse"; "traincgb" "mae"; "traincgb" "sse"; "traincgb" "sae"; "traincgb" "crossentropy"];
tab_cgf = ["traincgf" "mse"; "traincgf" "mae"; "traincgf" "sse"; "traincgf" "sae"; "traincgf" "crossentropy"];
tab_cgp = ["traincgp" "mse"; "traincgp" "mae"; "traincgp" "sse"; "traincgp" "sae"; "traincgp" "crossentropy"];
tab_gdx = ["traingdx" "mse"; "traingdx" "mae"; "traingdx" "sse"; "traingdx" "sae"; "traingdx" "crossentropy"];
tab_gdm = ["traingdm" "mse"; "traingdm" "mae"; "traingdm" "sse"; "traingdm" "sae"; "traingdm" "crossentropy"];
tab_bfg = ["trainbfg" "mse"; "trainbfg" "mae"; "trainbfg" "sse"; "trainbfg" "sae"; "trainbfg" "crossentropy"];
tab_onss = ["trainoss" "mse"; "trainoss" "mae"; "trainoss" "sse"; "trainoss" "sae"; "trainoss" "crossentropy"];

%tabtreino = [tab_gd tab_rp tab_scg tab_cgb tab_cgf tab_cgp tab_gdx tab_gdm tab_bfg tab_onss];
tabtreino = [tab_gd tab_rp tab_scg tab_cgb tab_cgf tab_cgp tab_gdx tab_gdm tab_bfg tab_onss];

trainFcn = tabtreino(2,1);
net.performFcn = tabtreino(2,2);

net = feedforwardnet(nn,trainFcn);
net.trainParam.lr = n;



net = configure(net,x,t);

net.plotFcns = {'plotperform','plottrainstate','ploterrhist','plotconfusion', 'plotroc'};

[net,tr] = train(net,x,t);
