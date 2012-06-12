%% TUTORIAL OF KERNEL DENSITY ESTIMATION BY SSKERNEL.M AND SSVKERNEL.M
clear all; close all; disp('Start');
figure('Position',[1000 100 800 800] );
%%
% The algorithm can be used for a general purpose density estimation. 
% [y,t,optw,W,C,confb95,yb] = sskernel(x,t,W)

addpath('../data/britten/');
load ../data/britten/mat/e025%j102;%e025%j102%e025%w052
%e025
%j102%e025%%e023%j102%e025 %j102%w213%e025%e093%e023%e024 25 28

c = 51.2;%3.2%6.4%36.2%18.1;
x = [];
%for i = 1: length(info)
%    if (info{i}(3) == c);
%        x = [x [t{i}]*0.001]; 
%    end
%end
x = [t{1:end}]*0.001; clear t;

%% Create samples
% First, let's create a sample data set. 
% Here `x' is a vector containing N samples, following a specified
% distribution. 
if 1
N = 500;                       %Number of samples
x = 0.5-.5*log(rand(1,N));     %Exponential
%x = 1.5+0.5*log(rand(1,N));     %Exponential
%x = [x 0.2-0.5*log(rand(1,N))];     %Exponential
%x = 1.5+0.5*log(rand(1,N));     %Exponential
%x = 0.5+gamrnd(0.5,3,1,N);
%x = 0.5+1*rand(1,N);           %Square
%x = 1 + 0.1*randn(1,N);      %Normal

%x = [x 0.5 + 0.01*randn(1,N/10)];      %Normal
%x = [x 0.6 + 0.01*randn(1,N/10)];      %Normal

x = [x 0.4 + 0.03*randn(1,N/5)];   
x = [x 1.2 + 0.03*randn(1,N/10)];   
x = [x 2*rand(1,round(N/5))];
x = [x 2*rand(1,round(10))];
end

dt = 0.001;
x = round(x/dt)*dt;
% Let's plot the data using a histogram. Here to see the noisy data 
% data structure, we use the bin-width 5 times smaller than an optimal 
% bin-width selected by histogram optimization method, 'sshist(x)'. 
subplot(3,1,1:2); hold on; 
edges = linspace(min(x),max(x),400);5*sshist(x)
b = histc(x,edges); bar(edges,b/sum(b)/min(diff(edges)),1);
h = findobj(gca,'Type','patch'); 
set(h,'FaceColor',.7*[1 1 1],'EdgeColor',0.8*[1 1 1]);

%% Create a vector of estimation points
% Now, we estimate the underlying density from the samples.  
L = 500;%5*sshist(x)%1000;                   %number of estimation points
t = linspace(0,2,L);        %points at which density estimation is made
                            %points must be equi-distant.
%t = linspace(0.8,1.2,L);
%% Kernel Density Estimation by a Fixed Bandwidth
% To obtain a kernel estimated density with a fixed kernel bandwidth, type
[yf,tf,optw,w,c] = sskernel(x,t);

%tic; sskernel_orig(x), toc;

% yf is the estimated density with the optimal bandwidth optw. The
% estimation was given at the points in tf, which was automatically
% determined from the data.

% Display the results.
subplot(3,1,1:2); hold on; 
plot(tf,yf,'b-','LineWidth',1.00);
set(gca,'XLim',[min(t) max(t)]);
subplot(3,1,3); hold on; plot(t,optw*ones(1,L),'k-','LineWidth',1.00);
set(gca,'XLim',[min(t) max(t)]);
drawnow;

% If the third input argument, W, is a vector, the method search the
% optimal bandwidth from elements of the vector, W. Without the third
% argument, the method search an optimal bandwidth by a golden section
% section search. We can looked at the estimated error of the bandwidths as
% follows. 
W = linspace(0.1*optw,5*optw,100);
[~,~,optw1,w1,c1] = sskernel(x,t);
[~,~,optw2,w2,c2] = sskernel(x,t,W);
%figure; plot(w2,c2,'k.-',w1,c1,'r.-'); axis square;

% Using a scalar value as W, we obtain the 
% The code utilizes the FFT algorithm. It is much faster (>x50) than 
% the bulit-in ksdensity function. 
disp('sskernel'); tic; [yss,tss,optw] = sskernel(x,t,optw); toc;
disp('ksdensity'); tic; yks = ksdensity(x,t,'width',optw); toc; 
% The method is usually faster than ksdensity even if the bandwidth is
% optimized.
disp('sskernel optimization'); tic; [~,~,~] = sskernel(x,t); toc;


%% Locally Adaptive Kernel Density Estimation
%t = linspace(0.2,0.8,L);
%t = linspace(0.3,0.6,L);
% The function ssvkernel returns the kerenel density estimate using
% bandwidths locally adaptive to the data.
tic; [yv,tv,optwv,gs,c] = ssvkernel(x,t); toc;
gs(end)
% The locally adaptive bandwidth at time tv is written in optwv.
subplot(3,1,1:2); hold on; plot(tv,yv,'r-','LineWidth',1.0);
subplot(3,1,3); hold on; plot(tv,optwv,'r','LineWidth',1.0);
drawnow;
% NOTE:
% If L is too small, the locally adaptive method may oversmooth the data.
% However the larger L is, the longer it takes to optimize it. 
% It is recommended to make L smaller than 1000.

% Figure settings
subplot(3,1,1:2);
%ax = legend('hist','fixed','variable','Location','Best'); 
%legend(ax,'boxoff'); grid on; ylabel('density');

subplot(3,1,3);
ax = legend('fix','variable','Location','Best');
h = findobj(gca,'Type','patch'); 
set(h,'FaceColor',.7*[1 1 1],'EdgeColor',0.8*[1 1 1]);
legend(ax,'boxoff'); grid on; ylabel('bandwidth');
set(gca,'XLim',[min(t) max(t)]);
drawnow;

%% BARS
addpath('../3rd/barsP_matlab');
edges = linspace(min(t),max(t),400);
dedges = edges(2)-edges(1);
hc = histc(x,edges)';
%fit1 = barsP(hc,[0 2],60);

bp = defaultParams;
bp.prior_id = 'POISSON';
bp.dparams = 4;
%fit2 = barsP(histc(x,edges)',[0 2],60);
fit2 = barsP(histc(x,edges)',[0 2],60);

subplot(3,1,1:2); 
%plot(edges,fit1.mode / sum(hc) / dedges);
plot(edges,fit2.mode / sum(hc) / dedges,'g');

% The type of above samples is double-precision. Often real data
% have much lower precision. We can introduce the sampling resolution of
% data, dt, as 
%dt = 0.001;
%x = round(x/dt)*dt;

%[~,~,dt] = find( sort(diff(sort(x))),1,'first')
%x = x + dt*(rand(1,length(x))-0.5); %jittering


break


%%

%%
%subplot(3,1,1:2); [~,tv,~,~,~,conf,yb]=sskernel(x,t);
%plot(tv,mean(yb),'k'); 
%sskernel(x,t);
%plot(tf,yf,'k:');

subplot(3,1,1:2); [~,tv,~,~,~,conf,yb]=ssvkernel(x,t,'Visible');



break
%%
hold off;
for i =1: 100
    idx = ceil(rand(1,N)*N);
    xb(i,:) = x(idx);
    subplot(2,1,1); hist(x,30,'r'); 
    subplot(2,1,2); hist(xb(i,:),30); hold off;
    [hc(i,:) n] = hist(xb(i,:),30);
    drawnow;
end
subplot(2,1,1); hold on; plot(n,mean(hc),'r.'); 
%% 
cla;
x = randn(1,1e2);
x = 0;
t= linspace(-1,1,1000); dt = t(2)-t(1);
hc = histc(x,t-dt/2); bar(t,hc);

w = 0.1;
%ssvkernel(x,t,w);
[y,t,w,c,~,conf,yb]=sskernel(x,0,0.1);
%[y,t,~,~,~,conf,yb]=ssvkernel(x,t,w);
%[y,t,~,~,~,conf,yb]=ssvkernel(x,t);
plot(t,y,'r'); 
plot(t,mean(yb),'g--');
plot(t,median(yb),'y:');
plot(t,conf','k');

hc = histc(x,t-dt/2); bar(t,hc);

set(gca,'XLim',[min(t) max(t)]);

break

%% stiffness
[yv,tv,optwv,gs,Cv] = ssvkernel(x,t);

figure;
subplot(3,1,1); plot(1:length(Cv),Cv,'k.-');
subplot(3,1,2); plot(1:length(gs),gs,'k.-');
subplot(3,1,3); plot(gs,Cv,'g.-'); 
set(gca,'XLim',[0 1]);

break
%% Confidence
sskernel(x,t);
% Without output arguments, the function displays the density estimate with
% 95% bootstrap confidence intervals.

x_ab = x( logical((x >= min(t)) .*(x <= max(t))) ) ;
[~,~,dt] = find( sort(diff(sort(x_ab))),1,'first');


%%
clear all;
figure; hold on;
t = linspace(0,.5,5000);
N = 10;                       %Number of samples
x = 0.5-0.5*log(rand(1,N));
x = [1];

%%
[optw,C,W] = sskernel_orig(x,logspace(-2,1,50));
plot(W,C,'k'); plot(optw,C(W==optw),'k*');

[ygs,~,optw,wf,Cf] = sskernel(x,t,W);
hold on; plot(wf,Cf,'r-');
plot(optw,Cf(wf==optw),'r*');



%%
figure;
dt = t(2)-t(1);
[ygs,~,~,wf,Cf] = sskernel(x,t);
hold on; plot(wf,Cf,'r.-');
plot(wf(end),Cf(end),'r*');


[optw,wf,Cf] = sskernel_orig(x,t);
plot(wf,Cf,'k');

W = logspace(log10(2*dt),log10(max(x)-min(x)),1e3);
W = linspace((2*dt),0.1*(max(x)-min(x)),1e3);
[ygl,~,~,wfg,Cfg] = sskernel(x,t,W);
plot(wfg,Cfg,'k-');
[ygl2,~,~,wfg,Cfg] = sskernel(x,t,wf(end));
plot(wfg,Cfg,'ko');

figure; hold on; 
plot(t,ygs,'k',t,ygl,'r--');
plot(t,ygl2,'g--');


%%
figure; 
semilogx(gs,Cv,'g.-');

subplot(2,1,2); 
hold on; %plot(t,y_hist);
%plot(t,yvs'/N,'k');
plot(t,y,'b-');
plot(t,yv,'g-');
legend('global','variable');

    %x_ab = x( find((x >= min(t)) .*(x <= max(t))) ) ;
    %[~,~,dt2] = find( sort(diff(sort(x_ab))),1,'first');
    
    %dt = max(dt1,dt2);
    
%%
dt = 0.001;
[yg,~,optWg] = sskernel(x,t);
dt = t(2)-t(1)
%%



