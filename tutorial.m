%% TUTORIAL OF KERNEL DENSITY ESTIMATION BY SSKERNEL.M AND SSVKERNEL.M
clear all; close all;

%% Create samples
% First, let's create a sample data set. 
% Here `x' is a vector containing N samples, following a specified
% distribution. 

N = 500;                        %Number of samples
%x = 0.5-.5*log(rand(1,N));     %Exponential
%x = 0.5+1*rand(1,N);           %Square

%x = 1 + 0.1*randn(1,N);        %Normal

if 1
x = [0.5-.5*log(rand(1,N))...   %Complex
    0.4 + 0.03*randn(1,N/5) ...
    1.2 + 0.03*randn(1,N/10)];
end

%% Plot a histogram
% Let's plot the data using a histogram. Here to see the noisy data 
% data structure, we use the bin-width 5 times smaller than an optimal 
% bin-width selected by histogram optimization method, 'sshist(x)'. 
subplot(3,1,1:2); hold on; 
edges = linspace(0,2,200);
b = histc(x,edges); bar(edges,b/sum(b)/min(diff(edges)),1);
h = findobj(gca,'Type','patch'); 
set(h,'FaceColor',.7*[1 1 1],'EdgeColor',0.8*[1 1 1]);
set(gca,'XTickLabel',[]);

%% Create a vector of estimation points
% Now, we estimate the underlying density from the samples.  
L = 2000;                    %number of estimation points
t = linspace(0,2,L);        %points at which density estimation is made
                            %points must be equi-distant.

%% Kernel Density Estimation by a Fixed Bandwidth
% To obtain a kernel estimated density with a fixed kernel bandwidth, type
[yf,tf,optw,w,c] = sskernel(x,t);

% yf is the estimated density with the optimal bandwidth optw. The
% estimation was given at the points in tf, which was automatically
% determined from the data. 

% Display the results.
subplot(3,1,1:2);
hold on; plot(tf,yf,'b-','LineWidth',2);
set(gca,'XLim',[min(t) max(t)]);
ylabel('density');
subplot(3,1,3); hold on; 
plot(t,optw*ones(1,L),'b-','LineWidth',2);
set(gca,'XLim',[min(t) max(t)]);
ylabel('bandwidth');
drawnow;

%% Locally Adaptive Kernel Density Estimation
%t = linspace(0.2,0.8,L);
%t = linspace(0.3,0.6,L);
% The function ssvkernel returns the kerenel density estimate using
% bandwidths locally adaptive to the data.
tic; [yv,tv,optwv,gs,cs] = ssvkernel(x,t); toc;

% The locally adaptive bandwidth at time tv is written in optwv.
subplot(3,1,1:2); hold on; plot(tv,yv,'r-','LineWidth',2);
subplot(3,1,3); hold on; plot(tv,optwv,'r','LineWidth',2);
drawnow;

%% Speed of the fixed kernel density estimate
% The code utilizes the FFT algorithm. It is much faster (>x50) than 
% the bulit-in ksdensity function. 
disp('sskernel'); tic; [yss,tss,optw] = sskernel(x,t,optw); toc;
disp('ksdensity'); tic; yks = ksdensity(x,t,'width',optw); toc; 
% The method should be faster than ksdensity even if the bandwidth is
% optimized.
disp('sskernel optimization'); tic; [yv,tv,wb] = sskernel(x,t); toc;

% Note on the third input argument.
% If the third input argument, W, is a vector, the method search the
% optimal bandwidth from elements of the vector, W. Without the third
% argument, the method search an optimal bandwidth by a golden section
% section search. We can looked at the estimated error of the bandwidths as
% follows. 
W = linspace(0.1*optw,5*optw,100);
[yv1,tv1,optw1,w1,c1] = sskernel(x,t);
[yv2,tv2,optw2,w2,c2] = sskernel(x,t,W);
%figure; plot(w2,c2,'k.-',w1,c1,'r.-'); axis square;


%% (Figure settings)
subplot(3,1,1:2);
ax = legend('histogram','fixed','locally adaptive','Location','Best'); 
legend(ax,'boxoff'); grid on; ylabel('density');

subplot(3,1,3);
ax = legend('fix','locally adaptive','Location','Best');
h = findobj(gca,'Type','patch'); 
set(h,'FaceColor',.7*[1 1 1],'EdgeColor',0.8*[1 1 1]);
legend(ax,'boxoff'); grid on; ylabel('bandwidth');
set(gca,'XLim',[min(t) max(t)]);
drawnow;
