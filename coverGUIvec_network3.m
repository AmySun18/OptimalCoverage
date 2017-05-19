
function coverGUIvec_network2()
% The GUI is a simulation of coverage control problem
% in a square mission space without obstacles.
% This is an optimization problem.
% The objective function is H(s)=integral of R(x)P(x,s)
% Algorithm is gradient-based algorithm and the technique
% to calculate the derivatites is discretization.
% Written By: Xinmiao Sun

%___________________________
% clear variables, command window, and figure window

clear
clc
close all;


%% 

%_______________________________
% create figure window

figure_h = figure(...
    'Visible','off',...
    'Units','normalized',...
    'Position',[0.2 0.2 0.6 0.6],...
    'Name','coverage control GUI',...
    'Color',[0.8 0.8 0.8]...
    );

movegui(figure_h,'center')

% clear figure window
clf

%_______________________________
% create plot objects

% plots nodes position in the mission space
nodesplot_h = axes(...
    'Visible','off',...
    'Units','normalized',...
    'Parent',figure_h,...
    'Position',[0.05 0.35 0.6 0.55]...
    );


%_________________________________
% create parameter objects

% parameters button group

parameters_h = uibuttongroup(...
    'Visible','off',...
    'Units','normalized',...
    'Parent',figure_h,...
    'Position',[0.05 0.05 0.6 0.22],...
    'BackgroundColor',[0.87 0.87 0.87],...
    'FontSize',10,...
    'TitlePosition','centertop'...
    );

% sensor number drop-down menu

    % drop-down menu label
    numctxt_h = uicontrol(...
        'Style','text',...
        'Visible','off',...
        'Units','normalized',...
        'Parent',parameters_h,...
        'BackgroundColor',[0.87 0.87 0.87],...
        'Position',[0.1 0.6 0.15 0.3],...
        'FontSize',9,...
        'String','number of sensors:'...
        );

    % drop-down menu
   
    numsensor=1:15;
    
    numcddm_h = uicontrol(...
        'Style','popupmenu',...
        'Visible','off',...
        'Units','normalized',...
        'Parent',parameters_h,...
        'Position',[0.12 0.2 0.1 0.35],...
        'FontSize',9,...
        'String',numsensor,...
        'Callback',@numpopfn...
        );

% specific sensor range edit box

    % edit box label
    senrangetxt_h = uicontrol(...
        'Style','text',...
        'Visible','off',...
        'Units','normalized',...
        'Parent',parameters_h,...
        'BackgroundColor',[0.87 0.87 0.87],...
        'Position',[0.32 0.6 0.1 0.3],...
        'FontSize',9,...
        'String','Sensing range:'...
        );

       % edit box
   senrangedit_h = uicontrol(...
        'Style','edit',...
        'Visible','off',...
        'Units','normalized',...
        'Parent',parameters_h,...
        'Position',[0.32 0.3 0.1 0.28],...
        'FontSize',9,...
        'String','80'...
        );

% Leader position edit box

    % edit box label
    leadertxt_h = uicontrol(...
        'Style','text',...
        'Visible','off',...
        'Units','normalized',...
        'Parent',parameters_h,...
        'BackgroundColor',[0.87 0.87 0.87],...
        'Position',[0.48 0.6 0.25 0.3],...
        'FontSize',9,...
        'String','Leader Position(Format: x y):'...
        );
      leaderedit_h = uicontrol(...
        'Style','edit',...
        'Visible','off',...
        'Units','normalized',...
        'Parent',parameters_h,...
        'Position',[0.52 0.3 0.15 0.28],...
        'FontSize',9,...
        'String','10 10'...
        );
    
% specific misson space edit box
        % edit box label
    missiontxt_h = uicontrol(...
        'Style','text',...
        'Visible','off',...
        'Units','normalized',...
        'Parent',parameters_h,...
        'BackgroundColor',[0.87 0.87 0.87],...
        'Position',[0.76 0.6 0.2 0.3],...
        'FontSize',9,...
        'String','The mission space area(Format: x y):'...
        );

    % edit box
   missionedit_h = uicontrol(...
        'Style','edit',...
        'Visible','off',...
        'Units','normalized',...
        'Parent',parameters_h,...
        'Position',[0.76 0.3 0.2 0.28],...
        'FontSize',9,...
        'String','60 50'...
        );



%_________________________________
% create control objects

% Control button group

    control_h = uibuttongroup(...
    'Visible','off',...
    'Units','normalized',...
    'Parent',figure_h,...
    'Position',[0.7 0.05 0.25 0.22],...
    'BackgroundColor',[0.87 0.87 0.87],...
    'TitlePosition','centertop'...
    );

    % button group label
    contitletxt_h = uicontrol(...
        'Style','text',...
        'Visible','off',...
        'Units','normalized',...
        'Parent',control_h,...
        'BackgroundColor',[0.87 0.87 0.87],...
        'Position',[0.1 0.75 0.8 0.15],...
        'FontSize',10,...
        'FontWeight','bold',...
        'String','Control Button:'...
        );

    
    % push button
    
    % Show button
     showbn_h = uicontrol(...
    'Style','pushbutton',...
    'Visible','off',...
    'Units','normalized',...
    'Parent',control_h,...
    'Position',[0.25 0.58 0.5 0.13],...
    'FontSize',10,...
    'String','Show',...
    'Callback',@showbuttonfn...
    );
    
    
    % Start button
    startbn_h = uicontrol(...
    'Style','pushbutton',...
    'Visible','off',...
    'Units','normalized',...
    'Parent',control_h,...
    'Position',[0.25 0.43 0.5 0.13],...
    'FontSize',10,...
    'String','Start',...
    'Callback',@startbuttonfn...
    );
    
     % Opti Comm button
    optiCommbn_h = uicontrol(...
    'Style','pushbutton',...
    'Visible','off',...
    'Units','normalized',...
    'Parent',control_h,...
    'Position',[0.25 0.28 0.5 0.13],...
    'FontSize',10,...
    'String','Opt. Comm.',...
    'Callback',@optiCommbuttonfn...
    );

    % Stop button
     stopbn_h = uicontrol(...
    'Style','pushbutton',...
    'Visible','off',...
    'Units','normalized',...
    'Parent',control_h,...
    'Position',[0.25 0.1 0.5 0.13],...
    'FontSize',10,...
    'String','Stop',...
    'Callback',@stopbuttonfn...
    );
    
%     % instruction button
%    infobn_h = uicontrol(...
%     'Style','pushbutton',...
%     'Visible','off',...
%     'Units','normalized',...
%     'Parent',figure_h,...
%     'Position',[0.72 0.3 0.2 0.1],...
%     'FontSize',12,...
%     'ForegroundColor','red',...
%     'String','Instruction',...
%     'Callback',@infobuttonfn...
%     );

        % edit box label
    objectivetxt_h = uicontrol(...
        'Style','text',...
        'Visible','off',...
        'Units','normalized',...
        'Parent',figure_h,...
        'BackgroundColor',[0.87 0.87 0.87],...
        'Position',[0.72 0.37 0.2 0.03],...
        'FontSize',10,...
        'String','Objective value:'...
        );

    % edit box
   objectiveedit_h = uicontrol(...
        'Style','edit',...
        'Visible','off',...
        'Units','normalized',...
        'Parent',figure_h,...
        'Position',[0.72 0.3 0.2 0.05],...
        'FontSize',9 ...
        );

%_________________________________
% create uiTable object

    % table title
    tabletxt_h = uicontrol(...
        'Style','text',...
        'Visible','off',...
        'Units','normalized',...
        'Parent',figure_h,...
        'BackgroundColor',[0.8 0.8 0.8],...
        'Position',[0.725 0.855 0.2 0.1],...
        'FontSize',11,...
        'FontWeight','bold',...
        'String','Sensor positions:'...
        );
    
   
    

    % table
    columneditable=[true true];
    rownames={'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'};
    columnformat = {'numeric','numeric'};
    table_h = uitable(...
        'Visible','off',...
        'Units','normalized',...
        'Parent',figure_h,...
        'Position',[0.7 0.44 0.25 0.45],...
        'ColumnName',{'x ','y'},...
        'RowName',rownames,...
        'ColumnFormat', columnformat,...
        'ColumnEditable',columneditable,... 
        'Data',[0.1 0.1]...
        );


    


% ________________________________
% make all visible

set([figure_h nodesplot_h  parameters_h numctxt_h ...
    numcddm_h senrangetxt_h senrangedit_h leadertxt_h leaderedit_h missiontxt_h missionedit_h ...
    control_h  contitletxt_h startbn_h optiCommbn_h showbn_h...
    stopbn_h objectivetxt_h objectiveedit_h tabletxt_h table_h],...
    'Visible','on'...
    );
        Mispace=str2num(get(missionedit_h,'String'));
        SensingRange=str2num(get(senrangedit_h,'String'));
        Num_nodes=get(numcddm_h,'Value');
        Data=get(table_h,'Data');
        S=Data(1:Num_nodes,:);
        %color=rand(Num_nodes,3);
        %scatter(nodesplot_h,S(:,1),S(:,2),40,color);
        scatter(nodesplot_h,S(:,1),S(:,2),100,'b');
        axis([0 Mispace(1) 0 Mispace(2)]);
        title('The moving nodes trajectories');
     

%%  Callback Functions

% Instruction 
    function infobuttonfn(~,~)
        
       open('Instruction.txt')
       
    end
% number of sensor nodes popup 

    function numpopfn(source,eventdata)
        
        % Get the string value that user choose
        num=get(numcddm_h,'Value');
%         remindtxt_h = uicontrol(...
%         'Style','text',...
%         'Visible','off',...
%         'Units','normalized',...
%         'Parent',figure_h,...
%         'BackgroundColor',[0.8 0.8 0.8],...
%         'Position',[0.7 0.8 0.25 0.1],...
%         'FontSize',12,...
%         'FontWeight','bold',...
%         'ForegroundColor','red',...
%         'String',sprintf('Reminder: Please enter the initial positions for sensors 1-%d  in the table:',num)...
%         );
%         set(remindtxt_h,'Visible','on');
            S=zeros(num,2);
            S0=[1:num;1:num]';
            S=S+S0*2;
            set(table_h,'Data',S)

              
    end
% Show button

    function showbuttonfn(~,~)
        
        Mispace=str2num(get(missionedit_h,'String'));
        SensingRange=str2num(get(senrangedit_h,'String'));
        Num_nodes=get(numcddm_h,'Value');
        Data=get(table_h,'Data');
        S=Data(1:Num_nodes,:);
        %color=rand(Num_nodes,3);
        %scatter(nodesplot_h,S(:,1),S(:,2),40,color);
        if any(Mispace<0)
            error('Your entered Mispace is not in the right range.please enter it again!');            
        end
        if SensingRange<=0
            error('Error! The sensor range should be greater than 0.');   
        end
        Num_nodes=get(numcddm_h,'Value');
        check=ones(Num_nodes,1)*Mispace;
        Data=get(table_h,'Data');
        S=Data(1:Num_nodes,:);
        if any(any(S>check))
            error('Please check your starting points and make sure that they are in the Mission space.')
        end
        
       s0=str2num(get(leaderedit_h,'String'));
       s0x=s0(1);
       s0y=s0(2);
      % scatter(nodesplot_h,s0x,s0y,100,'b');
       drawx=[S(:,1); s0x];
       drawy=[S(:,2); s0y];        
       scatter(nodesplot_h,drawx,drawy,100,'b');
        axis([0 Mispace(1) 0 Mispace(2)]);
        title('The moving nodes trajectories');
         for i=1:Num_nodes
                text(S(i,1),S(i,2),sprintf('%d',i));
         end
            text(s0x,s0y,sprintf('L'));
        
            g=getObjectiveValue();
            set(objectiveedit_h,'String',num2str(g));
    end
% Start button
    function startbuttonfn(source,eventdata)
        % To plot the sensor trajectory
        
        Mispace=str2num(get(missionedit_h,'String'));
        if any(Mispace<0)
            error('Your entered Mispace is not in the right range.please enter it again!');
            
        end
        SensingRange=str2num(get(senrangedit_h,'String'));
        if SensingRange<=0
            error('The sensor range should be greater than 0.');
            
        end
        Num_nodes=get(numcddm_h,'Value');
        check=ones(Num_nodes,1)*Mispace;
        Data=get(table_h,'Data');
        S=Data(1:Num_nodes,:);
        if any(any(S>check))
            error('Please check your starting points and make sure that they are in the Mission space.')   
        end
        % Leader position
        so=get()
        
        % Initialization
        flagvalue=1;
        setappdata(startbn_h,'flag',flagvalue);
       
        
        % distance function and probability function
        
        % Convenience assumption: the event density function R(x)=1 
        % The probability that sensor node i detects an event occuring at x=(x1,x2)
        % is pi(x,Si)=poi*exp(-lamda*norm(x,Si)). If x is not in the region of
        % nodes i, pi(x,Si)=0;
        % I assume that they are same for simplification and poi=1, lamda=5.
        p0=1;
        lamda=0.06;
        %p=@(x,s) p0*exp(-lamda*d(x,s));
        %dp_dd=@(x,s) -lamda*p(x,s);
        
        %% Discretization using a grid

        % For node i, the region of coverage Omiga is a circle with radius D. After
        % discretization, the region is represented by a (2V+1)*(2V+1) grid.

        % Resolution
        Delta=1;
        V=floor(SensingRange/Delta);
        
        % coordinate transformation x=(x1,x2)-->XL=(u,v) by XL=(x-s)./Delta 
        flag=getappdata(startbn_h,'flag');
        stepchange=1;
       while flag 
           
            for i=1:Num_nodes
            
            % To find the neighbour of node i in the vector N
            N=zeros(1,Num_nodes-1);
            si=S(i,:);
            dis=mydis(S,si);
            N=find((dis<2*SensingRange).*(dis~=0));
           

            % for the nodes si, find its range
            
            Omiga_large=(Mispace-si)./Delta;
            Omiga_small=(-si)./Delta;
            m_large=min(floor(Omiga_large(1)),V);
            m_small=max(ceil(Omiga_small(1)),-V);
            j_large=min(floor(Omiga_large(2)),V);
            j_small=max(ceil(Omiga_small(2)),-V);
            x=m_small:m_large;
            y=j_small:j_large;
            [X,Y]=meshgrid(x,y);
            [row, column]=size(X);
            total=row*column;
            % Area is the local coordinates in si range 
            Area=[reshape(X,total,1) reshape(Y,total,1)];
            % X is the original coordinates in si range
            X=Area.*Delta+ones(total,1)*si;
            % B is the product term for each position
            B=ones(total,1);
                
            for k=1:length(N)
                sk=S(N(k),:);
                dis=mydis(X,sk);
                Pr=p0*exp(-lamda.*dis);
                Index=find(dis<SensingRange);
                B(Index)=B(Index).*(1-Pr(Index));
            end
            
            % Check whether leader is within the neighbors
            dis=mydis(s0,si);
            if(dis<2*SensingRange)
                Pr=p0*exp(-lamda.*dis);
                Index=find(dis<SensingRange);
                B(Index)=B(Index).*(1-Pr(Index));                
            end
                
                dH=zeros(total,2);
                Pl=p0*exp(-lamda.*mydis(X,si));
                Dp_ddl=-lamda.*Pl;
                den=sqrt(sum(Area.^2')');
                num=[-B.*Dp_ddl -B.*Dp_ddl].*Area;
                In=find(den~=0);
                dH(In,:)=num(In,:)./[den(In) den(In)];
                
                si=si+Num_nodes^1.1/stepchange^0.8*Delta^2*sum(dH);
                si=min(si,Mispace);
                S(i,:)=si;
                
                set(table_h,'Data',S);
                
                
            end
            g=getObjectiveValue();
            set(objectiveedit_h,'String',num2str(g));
            stepchange=stepchange+1;
            % scatter(nodesplot_h,S(:,1),S(:,2),100,color);
            scatter(nodesplot_h,S(:,1),S(:,2),100,'b');
            axis([0 Mispace(1) 0 Mispace(2)]);
            title(nodesplot_h,'The moving nodes trajectories',...
                'FontSize',15,'FontWeight','bold');
            %legend(nodesplot_h,'Node 1','Node 2','Node 3','Node 4',...
                %'Node 5','Node 6','Node 7','Node 8','Node 9','Node 10');
            for i=1:Num_nodes
                text(S(i,1),S(i,2),sprintf('%d',i));
            end
            drawnow
            flag=getappdata(startbn_h,'flag');
        end
        
    end

    function optiCommbuttonfn(source, eventdata)
        % X is the unknown vector
        % X=(s1x, s2x,..., s1y, s2y, sNy,y01,y21, yN1, y02,y12,y32,...yN2, ..., y0N,y1N,...,y(N-1)N) 
        % in total of 2N+N*N
        % Ascii code of C is 67; Ascii code of I is 73;
        clear Opt
        Num_nodes=get(numcddm_h,'Value');
        Mispace=str2num(get(missionedit_h,'String'));
        charC=char(67*ones(1,Num_nodes*2));
        charI=char(73*ones(1,(Num_nodes)*Num_nodes));
        xtype=strcat(charC,charI);
        %linear constraints
        %sum_j y0j=N
        %sum yij-sum_i>0 yji=1; i~=j for each j=1,...N
        %yii=0
        %yij(dij-C)<=0
        %dij>eps
        b1=Num_nodes;
        % equality constraints
        % for leader, all out flow is N, constraint a1      
        % for every node, flow in-flow out =1; constraint in aeq
        a1=zeros(1,2*Num_nodes+(Num_nodes)*Num_nodes);
        aeq=zeros(Num_nodes,2*Num_nodes+(Num_nodes)*Num_nodes);       
        %yii=0;
%        aself=zeros(Num_nodes,2*Num_nodes+(Num_nodes+1)*Num_nodes);
%         % inequality constraints
%         a=zeros(1,2*Num_nodes+(Num_nodes+1)^2);
        multiplayer=eye(Num_nodes);
        for j=1:Num_nodes    
            a1((j+1)*(Num_nodes)+1)= 1;
            aeq(j,(j+1)*(Num_nodes)+1:(j+1)*(Num_nodes)+Num_nodes)=ones(1,Num_nodes);
            for k=1:Num_nodes
                if(k<j)
                    aeq(j,(k+1)*Num_nodes+j) = -1;
                else if(k>j)
                    aeq(j,(k+1)*Num_nodes+j+1)=-1;
                    end
                end
            end
            
        end
       
        Aeq=[a1;aeq];
        beq=[b1;ones(Num_nodes,1)];
%         % linear <= constraint
%         A=-a;
%         b=-1;
        A=[];
        b=[];
        
        lb = [zeros(2*Num_nodes+(Num_nodes)*Num_nodes,1)]; 
        ub = [Mispace(1)*ones(Num_nodes,1); Mispace(2)*ones(Num_nodes,1);Num_nodes*ones((Num_nodes)*Num_nodes,1)];
       
%         cl=zeros((Num_nodes+1)*Num_nodes,1);
%         % Communication range Cm=10;
%         Cm=10;
%         cu=Cm*ones((Num_nodes+1)*Num_nodes,1);
            eps=0.5;

           cl = -Inf*ones((Num_nodes)*Num_nodes,1);
           cu = zeros((Num_nodes)*Num_nodes,1);
           


                 %x0 
          Data=get(table_h,'Data');
          S=Data(1:Num_nodes,:);
          S=reshape(S,1,2*Num_nodes);
          x0=zeros(1,2*Num_nodes+(Num_nodes)*Num_nodes);
          x0(1:2*Num_nodes)=S;
          %x01=1; x02=1;...x0N=1; others
          x0(2*Num_nodes+1:Num_nodes:end)=ones(1,Num_nodes); 
%           for j=1:Num_nodes
%               x0(Num_nodes+j*(Num_nodes+1)+j-1)=Num_nodes-j+1;
%           end
         x0=x0'

        Opt = opti('fun',@objectivefun,'nl',@nonlinconstraint,cl,cu,'eq',Aeq,beq,'ineq',A,b,'bounds',lb,ub,...
             'xtype',xtype)
            [xvalue,fval,exitflag,info] =solve(Opt,x0)
           

            
            
          
            s0=str2num(get(leaderedit_h,'String'));
            s0x=s0(1);
            s0y=s0(2);
            
            drawx=[xvalue(1:Num_nodes); s0x];
            drawy=[xvalue(Num_nodes+1:2*Num_nodes); s0y];        
            scatter(nodesplot_h,drawx,drawy,100,'b');
            axis([0 Mispace(1) 0 Mispace(2)]);
            title(nodesplot_h,'The moving nodes trajectories',...
                'FontSize',15,'FontWeight','bold');
            %legend(nodesplot_h,'Node 1','Node 2','Node 3','Node 4',...
                %'Node 5','Node 6','Node 7','Node 8','Node 9','Node 10');
            for i=1:Num_nodes
                text(xvalue(i,1),xvalue(i+Num_nodes,1),sprintf('%d',i));
            end
            
            text(s0x,s0y,sprintf('L'));
            
            Data=[xvalue(1:Num_nodes) xvalue(Num_nodes+1:2*Num_nodes)];
            set(table_h,'DAta',Data);
            obj=getObjectiveValue();
            set(objectiveedit_h, 'String', num2str(obj));
            
            
            
    end

   function h=nonlinconstraint(X)
        Num_nodes=get(numcddm_h,'Value');
        %Num_nodes=2;
        %X=[15 10 30 30 0 1 0 1 0 0 0 0 0];
        i=(0:Num_nodes)';
        j=(0:Num_nodes)';
        % xij 
       sx=X(1:Num_nodes);
       sy=X(Num_nodes+1:2*Num_nodes);
       [I J]=meshgrid(i,j);
       ConsX=reshape(I',(Num_nodes+1)^2,1);
       ConsY=reshape(J',(Num_nodes+1)^2,1);
       %%%
       s0=str2num(get(leaderedit_h,'String'));
       s0x=s0(1);
       s0y=s0(2);
       sx=[s0x; sx];  % from s0 to sN
       sy=[s0y; sy];
       distance=sqrt((sx(ConsX+1)-sx(ConsY+1)).^2+(sy(ConsX+1)-sy(ConsY+1)).^2)';
       %delete d00,d10,d20,dN0 first N+1
       distance = distance(Num_nodes+2:end);
       %h=distance.*sign(X((2*Num_nodes+1):end))';
       for j=1:Num_nodes
           distance((j-1)*Num_nodes+j+1)=[];
       end
       % Communication range Cm=10;
       Cm=10;
       product=(distance-Cm).*X((2*Num_nodes+1):end)';
       h=[product];
     
   end


    function g=getObjectiveValue()
            Mispace=str2num(get(missionedit_h,'String'));
            SensingRange=str2num(get(senrangedit_h,'String'));
            lamda=0.06;
            missionSpaceLargeX = Mispace(1);
            missionSpaceLargeY = Mispace(2);
            
            
            p0=1;
            Data=get(table_h,'Data');
            S=Data(1:Num_nodes,:);
            s0=str2num(get(leaderedit_h,'String'));
            s0x=s0(1);
            s0y=s0(2);
            
            sx=S(:,1);
            sy=S(:,2);
           sx=[s0x; sx];  % from s0 to sN
           sy=[s0y; sy];
            x=0.01:1:missionSpaceLargeX;
            y=0.01:1:missionSpaceLargeY;
            [MissonX MissonY]=meshgrid(x,y);
            [row column]=size(MissonX);
            total=row*column;
            % Area is the coordinates of the whole mission space range 
            Area=[ reshape(MissonX,total,1) reshape(MissonY,total,1)];
         
            % B is the product term for each position
            miss_prob=ones(total,1);
                
            for k=1:length(sx)
                sk=[sx(k) sy(k)];
                dis=mydis(Area,sk);
                Pr=p0*exp(-lamda.*dis);
                Index=find(dis<SensingRange);
                miss_prob(Index)=miss_prob(Index).*(1-Pr(Index));
            end
            
            probAtOnePoint = ones(total,1)-miss_prob;
            
            g=sum(probAtOnePoint);
            
            
    end

    function g=objectivefun(X)
        %When I call integral2,x is 14*14 matrix.       
%         Mispace=str2num(get(missionedit_h,'String'));
%         missionSpaceLargeX = Mispace(1);
%         missionSpaceLargeY = Mispace(2);
%         g = integral2(@(x,y)probfun(X,x,y),0,missionSpaceLargeX,0,missionSpaceLargeY);    
        
        %calculate the objective value by hand
            Mispace=str2num(get(missionedit_h,'String'));
            SensingRange=str2num(get(senrangedit_h,'String'));
            lamda=0.06;
            missionSpaceLargeX = Mispace(1);
            missionSpaceLargeY = Mispace(2);
            
            
            p0=1;
           sx=X(1:Num_nodes);
           sy=X(Num_nodes+1:2*Num_nodes);
            s0=str2num(get(leaderedit_h,'String'));
            s0x=s0(1);
            s0y=s0(2);
           sx=[s0x; sx];  % from s0 to sN
           sy=[s0y; sy];
            x=0.01:1:missionSpaceLargeX;
            y=0.01:1:missionSpaceLargeY;
            [MissonX MissonY]=meshgrid(x,y);
            [row column]=size(MissonX);
            total=row*column;
            % Area is the coordinates of the whole mission space range 
            Area=[ reshape(MissonX,total,1) reshape(MissonY,total,1)];
         
            % B is the product term for each position
            miss_prob=ones(total,1);
                
            for k=1:length(sx)
                sk=[sx(k) sy(k)];
                dis=mydis(Area,sk);
                Pr=p0*exp(-lamda.*dis);
                Index=find(dis<SensingRange);
                miss_prob(Index)=miss_prob(Index).*(1-Pr(Index));
            end
            
            probAtOnePoint = ones(total,1)-miss_prob;
            
            g=-sum(probAtOnePoint);
            
            
    end  
            
%     function f=probfun(X,x,y)      
%         Num_nodes=get(numcddm_h,'Value');
%         lamda=0.06;
%        sx=X(1:Num_nodes);
%        sy=X(Num_nodes+1:2*Num_nodes);
%        s0x=10;
%        s0y=10;
%        sx=[s0x; sx];  % from s0 to sN
%        sy=[s0y; sy];
%        
%         
%         missProb = 1;
%         SensingRange = 80;
%         for i=1:size(sx)
%             distance = sqrt((sx(i)-x).^2+(sy(i)-y).^2);
%             if(distance < SensingRange)
%               alpha = 1;
%             else
%                alpha = 0;
%             end
%             
%             missProb = missProb * ( 1 - alpha*exp(-lamda * distance));
%         end
%         f = 1-missProb;
%         
%     end

% stop button
    function stopbuttonfn(source,eventdata)
        
        setappdata(startbn_h,'flag',0);
        
    end

            

end