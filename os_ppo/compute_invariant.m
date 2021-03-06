clear
clc
sdpvar x y d e;

f1 = [x+0.05*y; 
    (3*y+0.05*((1-9*x^2)*3*y - 3*x + 20*(-0.124996553742897*(y*3)^3*(-1.0*(x*3) - 1.0)^3 - 0.374978142367584*(y*3)^3*(-1.0*(x*3) - 1.0)^2*(1.0*(x*3) + 2.0) - 1.49981516433091*(y*3)^3*(-1.0*(x*3) - 1.0)*(0.5*(x*3) + 1)^2 - 0.999761753500694*(y*3)^3*(0.5*(x*3) + 1)^3 + 0.749802845732891*(y*3)^2*(-1.0*(x*3) - 1.0)^3*(0.5*(y*3) + 1.0) + 2.24889255808011*(y*3)^2*(-1.0*(x*3) - 1.0)^2*(1.0*(x*3) + 2.0)*(0.5*(y*3) + 1.0) + 8.99170664381813*(y*3)^2*(-1.0*(x*3) - 1.0)*(0.5*(x*3) + 1)^2*(0.5*(y*3) + 1.0) + 5.98965079213474*(y*3)^2*(0.5*(x*3) + 1)^3*(0.5*(y*3) + 1.0) - 1.49614988149244*(y*3)*(-1.0*(x*3) - 1.0)^3*(0.5*(y*3) + 1.0)^2 - 4.47856721134867*(y*3)*(-1.0*(x*3) - 1.0)^2*(1.0*(x*3) + 2.0)*(0.5*(y*3) + 1.0)^2 - 17.8397936363347*(y*3)*(-1.0*(x*3) - 1.0)*(0.5*(x*3) + 1)^2*(0.5*(y*3) + 1.0)^2 - 11.8007731142503*(y*3)*(0.5*(x*3) + 1)^3*(0.5*(y*3) + 1.0)^2 + 0.974228471384197*(-1.0*(x*3) - 1.0)^3*(0.5*(y*3) + 1.0)^3 + 2.85976098124406*(-1.0*(x*3) - 1.0)^2*(1.0*(x*3) + 2.0)*(0.5*(y*3) + 1.0)^3 + 10.9912184177256*(-1.0*(x*3) - 1.0)*(0.5*(x*3) + 1)^2*(0.5*(y*3) + 1.0)^3 + 6.79873766651443*(0.5*(x*3) + 1)^3*(0.5*(y*3) + 1.0)^3)) + d)/3];
f2 = [x+0.05*y; 
    (3*y+0.05*((1-9*x^2)*3*y - 3*x + 20*(-0.0600050179426281*(y*3)^3*(-1.0*(x*3) - 1.0)^3 - 0.134921651736906*(y*3)^3*(-1.0*(x*3) - 1.0)^2*(1.0*(x*3) + 2.0) - 0.354845482999679*(y*3)^3*(-1.0*(x*3) - 1.0)*(0.5*(x*3) + 1)^2 - 0.429205624617324*(y*3)^3*(0.5*(x*3) + 1)^3 - 0.0982649670429386*(y*3)^2*(1 - 0.5*(y*3))*(-1.0*(x*3) - 1.0)^3 - 0.491116356707914*(y*3)^2*(1 - 0.5*(y*3))*(-1.0*(x*3) - 1.0)^2*(1.0*(x*3) + 2.0) - 1.81809874931358*(y*3)^2*(1 - 0.5*(y*3))*(-1.0*(x*3) - 1.0)*(0.5*(x*3) + 1)^2 - 0.574681013033679*(y*3)^2*(1 - 0.5*(y*3))*(0.5*(x*3) + 1)^3 + 0.828277267278371*(y*3)*(1 - 0.5*(y*3))^2*(-1.0*(x*3) - 1.0)^3 + 1.30249819393918*(y*3)*(1 - 0.5*(y*3))^2*(-1.0*(x*3) - 1.0)^2*(1.0*(x*3) + 2.0) + 3.16999917931219*(y*3)*(1 - 0.5*(y*3))^2*(-1.0*(x*3) - 1.0)*(0.5*(x*3) + 1)^2 + 1.05210273210839*(y*3)*(1 - 0.5*(y*3))^2*(0.5*(x*3) + 1)^3 + 0.974228471384197*(1 - 0.5*(y*3))^3*(-1.0*(x*3) - 1.0)^3 + 2.85976098124406*(1 - 0.5*(y*3))^3*(-1.0*(x*3) - 1.0)^2*(1.0*(x*3) + 2.0) + 10.9912184177256*(1 - 0.5*(y*3))^3*(-1.0*(x*3) - 1.0)*(0.5*(x*3) + 1)^2 + 6.79873766651443*(1 - 0.5*(y*3))^3*(0.5*(x*3) + 1)^3)) + d)/3];
f3 = [x+0.05*y; 
    (3*y+0.05*((1-9*x^2)*3*y - 3*x + 20*(0.124970219187587*(x*3)^3*(y*3)^3 - 0.748706349016842*(x*3)^3*(y*3)^2*(0.5*(y*3) + 1.0) + 1.47509663928129*(x*3)^3*(y*3)*(0.5*(y*3) + 1.0)^2 - 0.849842208314303*(x*3)^3*(0.5*(y*3) + 1.0)^3 - 0.374832213591299*(x*3)^2*(y*3)^3*(1.0*(x*3) + 1.0) + 2.2426424293506*(x*3)^2*(y*3)^2*(1.0*(x*3) + 1.0)*(0.5*(y*3) + 1.0) - 4.36110475696437*(x*3)^2*(y*3)*(1.0*(x*3) + 1.0)*(0.5*(y*3) + 1.0)^2 + 2.19332624693958*(x*3)^2*(1.0*(x*3) + 1.0)*(0.5*(y*3) + 1.0)^3 + 0.374684923599269*(x*3)*(y*3)^3*(1.0*(x*3) + 1.0)^2 - 2.23535925648821*(x*3)*(y*3)^2*(1.0*(x*3) + 1.0)^2*(0.5*(y*3) + 1.0) + 4.21589940460248*(x*3)*(y*3)*(1.0*(x*3) + 1.0)^2*(0.5*(y*3) + 1.0)^2 - 1.29345863902666*(x*3)*(1.0*(x*3) + 1.0)^2*(0.5*(y*3) + 1.0)^3 - 0.124792468482422*(y*3)^3*(1.0*(x*3) + 1.0)^3 + 0.738028123548326*(y*3)^2*(1.0*(x*3) + 1.0)^3*(0.5*(y*3) + 1.0) - 1.24237007529341*(y*3)*(1.0*(x*3) + 1.0)^3*(0.5*(y*3) + 1.0)^2 + 0.0714853160474517*(1.0*(x*3) + 1.0)^3*(0.5*(y*3) + 1.0)^3)) + d)/3];
f4 = [x+0.05*y; 
    (3*y+0.05*((1-9*x^2)*3*y - 3*x + 20*(0.0536507030771655*(x*3)^3*(y*3)^3 + 0.0718351266292099*(x*3)^3*(y*3)^2*(1 - 0.5*(y*3)) - 0.131512841513549*(x*3)^3*(y*3)*(1 - 0.5*(y*3))^2 - 0.849842208314303*(x*3)^3*(1 - 0.5*(y*3))^3 - 0.253832065822551*(x*3)^2*(y*3)^3*(1.0*(x*3) + 1.0) - 0.29426028926928*(x*3)^2*(y*3)^2*(1 - 0.5*(y*3))*(1.0*(x*3) + 1.0) - 0.0587814199671641*(x*3)^2*(y*3)*(1 - 0.5*(y*3))^2*(1.0*(x*3) + 1.0) + 2.19332624693958*(x*3)^2*(1 - 0.5*(y*3))^3*(1.0*(x*3) + 1.0) + 0.311206570234914*(x*3)*(y*3)^3*(1.0*(x*3) + 1.0)^2 + 0.275011819830898*(x*3)*(y*3)^2*(1 - 0.5*(y*3))*(1.0*(x*3) + 1.0)^2 + 0.230812552748116*(x*3)*(y*3)*(1 - 0.5*(y*3))^2*(1.0*(x*3) + 1.0)^2 - 1.29345863902666*(x*3)*(1 - 0.5*(y*3))^3*(1.0*(x*3) + 1.0)^2 - 0.114268163392215*(y*3)^3*(1.0*(x*3) + 1.0)^3 - 0.325410553876294*(y*3)^2*(1 - 0.5*(y*3))*(1.0*(x*3) + 1.0)^3 - 0.0920381682132145*(y*3)*(1 - 0.5*(y*3))^2*(1.0*(x*3) + 1.0)^3 + 0.0714853160474517*(1 - 0.5*(y*3))^3*(1.0*(x*3) + 1.0)^3)) + d)/3];
f5 = [x+0.05*y; 
    (3*y+0.05*((1-9*x^2)*3*y - 3*x + 20*(-0.120094250724849*(x*3)^3*(y*3)^3 + 0.51507448133777*(x*3)^3*(y*3)^2*(0.5*(y*3) + 1.0) - 0.168386350159562*(x*3)^3*(y*3)*(0.5*(y*3) + 1.0)^2 + 0.00992608650957551*(x*3)^3*(0.5*(y*3) + 1.0)^3 - 0.369912017036782*(x*3)^2*(y*3)^3*(1 - 1.0*(x*3)) + 1.95918787463977*(x*3)^2*(y*3)^2*(1 - 1.0*(x*3))*(0.5*(y*3) + 1.0) - 1.02934953732255*(x*3)^2*(y*3)*(1 - 1.0*(x*3))*(0.5*(y*3) + 1.0)^2 + 0.026771346000273*(x*3)^2*(1 - 1.0*(x*3))*(0.5*(y*3) + 1.0)^3 - 0.373255948561544*(x*3)*(y*3)^3*(1 - 1.0*(x*3))^2 + 2.14634869499568*(x*3)*(y*3)^2*(1 - 1.0*(x*3))^2*(0.5*(y*3) + 1.0) - 2.60490444204812*(x*3)*(y*3)*(1 - 1.0*(x*3))^2*(0.5*(y*3) + 1.0)^2 + 0.140458170324886*(x*3)*(1 - 1.0*(x*3))^2*(0.5*(y*3) + 1.0)^3 - 0.124792468482422*(y*3)^3*(1 - 1.0*(x*3))^3 + 0.738028123548326*(y*3)^2*(1 - 1.0*(x*3))^3*(0.5*(y*3) + 1.0) - 1.24237007529341*(y*3)*(1 - 1.0*(x*3))^3*(0.5*(y*3) + 1.0)^2 + 0.0714853160474517*(1 - 1.0*(x*3))^3*(0.5*(y*3) + 1.0)^3)) + d)/3];

f6 = [x+0.05*y; 
    (3*y+0.05*((1-9*x^2)*3*y - 3*x + 20*(-0.123746999757986*(x*3)^3*(y*3)^3 - 0.68889596662948*(x*3)^3*(y*3)^2*(1 - 0.5*(y*3)) - 0.701227589363739*(x*3)^3*(y*3)*(1 - 0.5*(y*3))^2 + 0.00992608650957551*(x*3)^3*(1 - 0.5*(y*3))^3 - 0.367250463551547*(x*3)^2*(y*3)^3*(1 - 1.0*(x*3)) - 1.87881197448617*(x*3)^2*(y*3)^2*(1 - 1.0*(x*3))*(1 - 0.5*(y*3)) - 0.68882120906208*(x*3)^2*(y*3)*(1 - 1.0*(x*3))*(1 - 0.5*(y*3))^2 + 0.026771346000273*(x*3)^2*(1 - 1.0*(x*3))*(1 - 0.5*(y*3))^3 - 0.359113780160417*(x*3)*(y*3)^3*(1 - 1.0*(x*3))^2 - 1.53047819705402*(x*3)*(y*3)^2*(1 - 1.0*(x*3))^2*(1 - 0.5*(y*3)) - 0.273920347751287*(x*3)*(y*3)*(1 - 1.0*(x*3))^2*(1 - 0.5*(y*3))^2 + 0.140458170324886*(x*3)*(1 - 1.0*(x*3))^2*(1 - 0.5*(y*3))^3 - 0.114268163392215*(y*3)^3*(1 - 1.0*(x*3))^3 - 0.325410553876294*(y*3)^2*(1 - 1.0*(x*3))^3*(1 - 0.5*(y*3)) - 0.0920381682132145*(y*3)*(1 - 1.0*(x*3))^3*(1 - 0.5*(y*3))^2 + 0.0714853160474517*(1 - 1.0*(x*3))^3*(1 - 0.5*(y*3))^3)) + d)/3];

f7 = [x+0.05*y; 
    (3*y+0.05*((1-9*x^2)*3*y - 3*x + 20*(-0.960754005798796*(y*3)^3*(1 - 0.5*(x*3))^3 - 1.34376270205323*(y*3)^3*(1 - 0.5*(x*3))^2*(1.0*(x*3) - 1.0) - 0.285357224966651*(y*3)^3*(2.0 - 1.0*(x*3))*(1.0*(x*3) - 1.0)^2 - 0.062207844201447*(y*3)^3*(1.0*(x*3) - 1.0)^3 + 4.12059585070216*(y*3)^2*(1 - 0.5*(x*3))^3*(0.5*(y*3) + 1.0) + 3.33858984363433*(y*3)^2*(1 - 0.5*(x*3))^2*(1.0*(x*3) - 1.0)*(0.5*(y*3) + 1.0) + 0.406548861123605*(y*3)^2*(2.0 - 1.0*(x*3))*(1.0*(x*3) - 1.0)^2*(0.5*(y*3) + 1.0) + 0.0337316583943885*(y*3)^2*(1.0*(x*3) - 1.0)^3*(0.5*(y*3) + 1.0) - 1.34709080127649*(y*3)*(1 - 0.5*(x*3))^3*(0.5*(y*3) + 1.0)^2 - 1.48234900446168*(y*3)*(1 - 0.5*(x*3))^2*(1.0*(x*3) - 1.0)*(0.5*(y*3) + 1.0)^2 - 0.079478360286649*(y*3)*(2.0 - 1.0*(x*3))*(1.0*(x*3) - 1.0)^2*(0.5*(y*3) + 1.0)^2 + 0.0707658273010061*(y*3)*(1.0*(x*3) - 1.0)^3*(0.5*(y*3) + 1.0)^2 + 0.079408692076604*(1 - 0.5*(x*3))^3*(0.5*(y*3) + 1.0)^3 + 0.829481968743917*(1 - 0.5*(x*3))^2*(1.0*(x*3) - 1.0)*(0.5*(y*3) + 1.0)^3 - 0.225998603115162*(2.0 - 1.0*(x*3))*(1.0*(x*3) - 1.0)^2*(0.5*(y*3) + 1.0)^3 - 0.341190805586866*(1.0*(x*3) - 1.0)^3*(0.5*(y*3) + 1.0)^3)) + d)/3];

f8 = [x+0.05*y; 
    (3*y+0.05*((1-9*x^2)*3*y - 3*x + 20*(-0.989975998063885*(y*3)^3*(1 - 0.5*(x*3))^3 - 1.49285556306035*(y*3)^3*(1 - 0.5*(x*3))^2*(1.0*(x*3) - 1.0) - 0.374155254843148*(y*3)^3*(2.0 - 1.0*(x*3))*(1.0*(x*3) - 1.0)^2 - 0.124866913420418*(y*3)^3*(1.0*(x*3) - 1.0)^3 - 5.51116773303584*(y*3)^2*(1 - 0.5*(x*3))^3*(1 - 0.5*(y*3)) - 8.6460358555581*(y*3)^2*(1 - 0.5*(x*3))^2*(1 - 0.5*(y*3))*(1.0*(x*3) - 1.0) - 2.20776226232803*(y*3)^2*(1 - 0.5*(y*3))*(2.0 - 1.0*(x*3))*(1.0*(x*3) - 1.0)^2 - 0.743316449629473*(y*3)^2*(1 - 0.5*(y*3))*(1.0*(x*3) - 1.0)^3 - 5.60982071490991*(y*3)*(1 - 0.5*(x*3))^3*(1 - 0.5*(y*3))^2 - 12.7159950673987*(y*3)*(1 - 0.5*(x*3))^2*(1 - 0.5*(y*3))^2*(1.0*(x*3) - 1.0) - 3.82136867060535*(y*3)*(1 - 0.5*(y*3))^2*(2.0 - 1.0*(x*3))*(1.0*(x*3) - 1.0)^2 - 1.37867799392542*(y*3)*(1 - 0.5*(y*3))^2*(1.0*(x*3) - 1.0)^3 + 0.079408692076604*(1 - 0.5*(x*3))^3*(1 - 0.5*(y*3))^3 + 0.829481968743917*(1 - 0.5*(x*3))^2*(1 - 0.5*(y*3))^3*(1.0*(x*3) - 1.0) - 0.225998603115162*(1 - 0.5*(y*3))^3*(2.0 - 1.0*(x*3))*(1.0*(x*3) - 1.0)^2 - 0.341190805586866*(1 - 0.5*(y*3))^3*(1.0*(x*3) - 1.0)^3)) + d)/3];


g_d=d^2;
% whole state space is [-2/3, 2/3]*[-2/3, 2/3], u(x) is constrained to ball 1 -(x^2 + y^2)>=0
H = 1 - x^2 - y^2;

% system constraints [-2/3, 2/3]*[-2/3, 2/3]
C0_0 = x^2 - 4/9;
C0_1 = y^2 - 4/9;

% f1 on constraints [-2/3, -1/3]*[-2/3, 0]
X1_0 = (x + 0.5)^2 - 1/36;
X1_1 = (y + 1/3)^2 - 1/9;

% f2 on constraints [-2/3, -1/3]*[0, 2/3]
X2_0 = (x + 0.5)^2 - 1/36;
X2_1 = (y - 1/3)^2 - 1/9;

% f3 on constraints [-1/3, 0]*[-2/3, 0]
X3_0 = (x + 1/6)^2 - 1/36; 
X3_1 = (y + 1/3)^2 - 1/9;

% f4 on constraints [-1/3, 0]*[0, 2/3]
X4_0 = (x + 1/6)^2 - 1/36; 
X4_1 = (y - 1/3)^2 - 1/9;

% f5 on constraints [0, 1/3]*[-2/3, 0]
X5_0 = (x - 1/6)^2 - 1/36; 
X5_1 = (y + 1/3)^2 - 1/9;

% f6 on constraints [0, 1/3]*[0, 2/3]
X6_0 = (x - 1/6)^2 - 1/36; 
X6_1 = (y - 1/3)^2 - 1/9;

% f7 on constraints [1/3, 2/3]*[-2/3, 0]
X7_0 = (x - 0.5)^2 - 1/36; 
X7_1 = (y + 1/3)^2 - 1/9;

% f8 on constraints [1/3, 2/3]*[0, 2/3]
X8_0 = (x - 0.5)^2 - 1/36; 
X8_1 = (y - 1/3)^2 - 1/9;



degree = 8;
x1=[x y];
x2 = [x y d];

[V,coe]= polynomial(x1,degree);

% h(x) = 1-x^2-y^2
% d= 6
% obj = (pi*(192*coe(1) + 48*coe(4) + 48*coe(6) + 24*coe(11) + 8*coe(13) + 24*coe(15) + 15*coe(22) + 3*coe(24) + 3*coe(26) + 15*coe(28)))/192;
% d=8
obj=(pi*(1920*coe(1) + 480*coe(4) + 480*coe(6) + 240*coe(11) + 80*coe(13) + 240*coe(15) + 150*coe(22) + 30*coe(24) + 30*coe(26) + 150*coe(28) + 105*coe(37) + 15*coe(39) + 9*coe(41) + 15*coe(43) + 105*coe(45)))/1920;
% d = 10
% obj=(pi*(7680*coe(1) + 1920*coe(4) + 1920*coe(6) + 960*coe(11) + 320*coe(13) + 960*coe(15) + 600*coe(22) + 120*coe(24) + 120*coe(26) + 600*coe(28) + 420*coe(37) + 60*coe(39) + 36*coe(41) + 60*coe(43) + 420*coe(45) + 315*coe(56) + 35*coe(58) + 15*coe(60) + 15*coe(62) + 35*coe(64) + 315*coe(66)))/7680;

V1=replace(V,x1,f1');
V2=replace(V,x1,f2');
V3=replace(V,x1,f3');
V4=replace(V,x1,f4');
V5=replace(V,x1,f5');
V6=replace(V,x1,f6');
V7=replace(V,x1,f7');
V8=replace(V,x1,f8');

degree1 = degree+6;

[s0,coe0]=polynomial(x2,degree1);
[s1,coe1]=polynomial(x2,degree1);
[s2,coe2]=polynomial(x2,degree1);
[s3,coe3]=polynomial(x2,degree1);
[s4,coe4]=polynomial(x2,degree1);
[s5,coe5]=polynomial(x2,degree1);
[s6,coe6]=polynomial(x2,degree1);
[s7,coe7]=polynomial(x2,degree1);
[s8,coe8]=polynomial(x2,degree1);
[s9,coe9]=polynomial(x2,degree1);
[s10,coe10]=polynomial(x2,degree1);
[s11,coe11]=polynomial(x2,degree1);

[s12,coe12]=polynomial(x2,degree1);
[s13,coe13]=polynomial(x2,degree1);
[s14,coe14]=polynomial(x2,degree1);
[s15,coe15]=polynomial(x2,degree1);
[s16,coe16]=polynomial(x2,degree1);
[s17,coe17]=polynomial(x2,degree1);
[s18,coe18]=polynomial(x2,degree1);
[s19,coe19]=polynomial(x2,degree1);
[s20,coe20]=polynomial(x2,degree1);
[s21,coe21]=polynomial(x2,degree1);
[s22,coe22]=polynomial(x2,degree1);
[s23,coe23]=polynomial(x2,degree1);

[s24,coe24]=polynomial(x2,degree1);
[s25,coe25]=polynomial(x2,degree1);
[s26,coe26]=polynomial(x2,degree1);
[s27,coe27]=polynomial(x2,degree1);
[s28,coe28]=polynomial(x2,degree1);
[s29,coe29]=polynomial(x2,degree1);
[s30,coe30]=polynomial(x2,degree1);
[s31,coe31]=polynomial(x2,degree1);

[s40,coe40]=polynomial(x1,degree1);
[s41,coe41]=polynomial(x1,degree1);

F=[sos(V-V1+s0*X1_0+s1*X1_1+s2*(g_d-0.04)-s3*H), sos(V-V2+s4*X2_0+s5*X2_1+s6*(g_d-0.04)-s7*H), sos(V-V3+s8*X3_0+s9*X3_1+s10*(g_d-0.04)-s11*H),
    sos(V-V4+s12*X4_0+s13*X4_1+s14*(g_d-0.04)-s15*H), sos(V-V5+s16*X5_0+s17*X5_1+s18*(g_d-0.04)-s19*H), sos(V-V6+s20*X6_0+s21*X6_1+s22*(g_d-0.04)-s23*H),
    sos(V-V7+s24*X7_0+s25*X7_1+s26*(g_d-0.04)-s27*H), sos(V-V8+s28*X8_0+s29*X8_1+s30*(g_d-0.04)-s31*H),
    sos((1+(C0_0)^2)*V-C0_0-s40*H), sos((1+(C0_1)^2)*V-C0_1-s41*H), sos(s0),sos(s1), sos(s2),sos(s3),sos(s4),sos(s5),sos(s6),sos(s7),sos(s8),sos(s9),sos(s10),sos(s11),sos(s12),sos(s13),
    sos(s14),sos(s15), sos(s16),sos(s17),sos(s18),sos(s19),sos(s20),sos(s21),sos(s22),sos(s23),sos(s24),sos(s25),sos(s26),sos(s27),sos(s28),sos(s29),sos(s30),sos(s31),sos(s40),sos(s41)];

ops = sdpsettings('solver','mosek','sos.newton',1,'sos.congruence',1);
diagnostics=solvesdp(F,obj,ops,[coe;coe0;coe1;coe2;coe3;coe4;coe5;coe6;coe7;coe8;coe9;coe10;coe11;coe12;coe13;coe14;coe15;
    coe16;coe17;coe18;coe19;coe20;coe21;coe22;coe23;coe24;coe25;coe26;coe27;coe28;coe29;coe30;coe31;coe40;coe41]);

if diagnostics.problem == 0
 disp('Solver thinks it is feasible')
elseif diagnostics.problem == 1
 disp('Solver thinks it is infeasible')
else
 disp('Something else happened')
end
v = monolist([x y],degree);
sdisplay(v'*double(coe));

% above code running result
%d=8
%-0.357383768019-0.0977143650762*x-0.0267949152538*y+0.662983677257*x^2-0.361192480951*x^2*y+0.410252670351*y^2+0.141324130177*y^3+0.349205035506*x^3-0.871585127725*x*y^3+0.384718470545*x^2*y^3+2.4959091816*x^3*y^3+0.439444426395*x*y^2-3.90897619843*x^2*y^2-1.56816467439*x^3*y^2+0.221712997844*x*y-0.316901787701*x^3*y+0.533815517192*x^4+1.43864453527*y^4-0.388073012871*x^5+1.05577057015*x^4*y-0.623034893834*x*y^4-0.110474858834*y^5-0.246895122972*x^6-0.466767461534*x^5*y+7.0796985825*x^4*y^2+4.19562725704*x^2*y^4+0.693863980004*x*y^5-1.0655262006*y^6+0.135886379069*x^7-0.689005901172*x^6*y+1.09589765935*x^5*y^2-0.720675129093*x^4*y^3+1.30156887551*x^3*y^4-0.135248531981*x^2*y^5+0.264868804654*x*y^6-0.00463431140825*y^7-0.16707732275*x^8+0.59845931213*x^7*y-3.03208198348*x^6*y^2-1.24626552468*x^5*y^3-8.77296493305*x^4*y^4-1.88879419528*x^3*y^5-0.634824553648*x^2*y^6+0.00199333871765*x*y^7-0.000108666568477*y^8