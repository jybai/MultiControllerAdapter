clear
clc
sdpvar x y d e;

% how to revise
% x, y belongs to [-2, -1]*[-2, 2]
f1 = [x+0.05*y; 
    (3*y+0.05*((1-9*x^2)*3*y - 3*x + 20*(0.425211076103333*(0.5 - 0.25*(3*y))*(-1.0*(3*x) - 1.0)^3*(0.5*(3*y) + 1)^2 + 1.02365428669482*(0.5 - 0.25*(3*y))*(-1.0*(3*x) - 1.0)^2*(1.0*(3*x) + 2.0)*(0.5*(3*y) + 1)^2 + 2.95504052725062*(0.5 - 0.25*(3*y))*(-1.0*(3*x) - 1.0)*(0.5*(3*x) + 1)^2*(0.5*(3*y) + 1)^2 + 1.15130885442421*(0.5 - 0.25*(3*y))*(0.5*(3*x) + 1)^3*(0.5*(3*y) + 1)^2 + 0.118905423011864*(1 - 0.5*(3*y))^3*(-1.0*(3*x) - 1.0)^3 + 0.348304683461052*(1 - 0.5*(3*y))^3*(-1.0*(3*x) - 1.0)^2*(1.0*(3*x) + 2.0) + 1.34236809633121*(1 - 0.5*(3*y))^3*(-1.0*(3*x) - 1.0)*(0.5*(3*x) + 1)^2 + 0.856536211071972*(1 - 0.5*(3*y))^3*(0.5*(3*x) + 1)^3 + 0.660880486810856*(1 - 0.5*(3*y))^2*(-1.0*(3*x) - 1.0)^3*(0.25*(3*y) + 0.5) + 1.89507514178296*(1 - 0.5*(3*y))^2*(-1.0*(3*x) - 1.0)^2*(1.0*(3*x) + 2.0)*(0.25*(3*y) + 0.5) + 7.1125340402736*(1 - 0.5*(3*y))^2*(-1.0*(3*x) - 1.0)*(0.5*(3*x) + 1)^2*(0.25*(3*y) + 0.5) + 4.32513165638667*(1 - 0.5*(3*y))^2*(0.5*(3*x) + 1)^3*(0.25*(3*y) + 0.5) + 0.00215343597608638*(-1.0*(3*x) - 1.0)^3*(0.5*(3*y) + 1)^3 - 0.0259029854064857*(-1.0*(3*x) - 1.0)^2*(1.0*(3*x) + 2.0)*(0.5*(3*y) + 1)^3 - 0.247595868986717*(-1.0*(3*x) - 1.0)*(0.5*(3*x) + 1)^2*(0.5*(3*y) + 1)^3 - 0.283330236206298*(0.5*(3*x) + 1)^3*(0.5*(3*y) + 1)^3)) + d)/3];
% x, y belongs to [-1, 1]*[-2, 2]
f2 = [x+0.05*y; 
    (3*y+0.05*((1-9*x^2)*3*y - 3*x + 20*(-0.243584378266395*(0.5 - 0.5*(3*x))*(0.5 - 0.25*(3*y))*((3*x) + 1)^2*(0.5*(3*y) + 1)^2 + 0.0556068616975541*(0.5 - 0.5*(3*x))*(1 - 0.5*(3*y))^3*((3*x) + 1)^2 + 0.0150306530769696*(0.5 - 0.5*(3*x))*(1 - 0.5*(3*y))^2*((3*x) + 1)^2*(0.25*(3*y) + 0.5) - 0.0647859069392533*(0.5 - 0.5*(3*x))*((3*x) + 1)^2*(0.5*(3*y) + 1)^3 + 0.0179892008503783*(0.5 - 0.25*(3*y))*(1 - (3*x))^3*(0.5*(3*y) + 1)^2 - 0.0566021333753245*(0.5 - 0.25*(3*y))*(1 - (3*x))^2*(0.5*(3*x) + 0.5)*(0.5*(3*y) + 1)^2 - 0.0572932821028275*(0.5 - 0.25*(3*y))*((3*x) + 1)^3*(0.5*(3*y) + 1)^2 + 0.0133833782979996*(1 - (3*x))^3*(1 - 0.5*(3*y))^3 + 0.0675801821310418*(1 - (3*x))^3*(1 - 0.5*(3*y))^2*(0.25*(3*y) + 0.5) - 0.0044270349407234*(1 - (3*x))^3*(0.5*(3*y) + 1)^3 + 0.071725194207449*(1 - (3*x))^2*(1 - 0.5*(3*y))^3*(0.5*(3*x) + 0.5) + 0.272500805218411*(1 - (3*x))^2*(1 - 0.5*(3*y))^2*(0.5*(3*x) + 0.5)*(0.25*(3*y) + 0.5) - 0.0530354952052034*(1 - (3*x))^2*(0.5*(3*x) + 0.5)*(0.5*(3*y) + 1)^3 + 0.00432268189630805*(1 - 0.5*(3*y))^3*((3*x) + 1)^3 - 0.0291911782061452*(1 - 0.5*(3*y))^2*((3*x) + 1)^3*(0.25*(3*y) + 0.5) - 0.0121985520699155*((3*x) + 1)^3*(0.5*(3*y) + 1)^3)) + d)/3];
% x, y belongs to [1, 2]*[-2, 2]
f3 = [x+0.05*y; 
    (3*y+0.05*((1-9*x^2)*3*y - 3*x + 20*(-3.66677005458096*(0.5 - 0.25*(3*y))*(1 - 0.5*(3*x))^3*(0.5*(3*y) + 1)^2 - 6.19193016324512*(0.5 - 0.25*(3*y))*(1 - 0.5*(3*x))^2*(1.0*(3*x) - 1.0)*(0.5*(3*y) + 1)^2 - 1.64930671669314*(0.5 - 0.25*(3*y))*(2.0 - 1.0*(3*x))*(1.0*(3*x) - 1.0)^2*(0.5*(3*y) + 1)^2 - 0.572970428326949*(0.5 - 0.25*(3*y))*(1.0*(3*x) - 1.0)^3*(0.5*(3*y) + 1)^2 + 0.276651641363715*(1 - 0.5*(3*x))^3*(1 - 0.5*(3*y))^3 - 1.86823540519329*(1 - 0.5*(3*x))^3*(1 - 0.5*(3*y))^2*(0.25*(3*y) + 0.5) - 0.780707332474591*(1 - 0.5*(3*x))^3*(0.5*(3*y) + 1)^3 + 0.209656119344319*(1 - 0.5*(3*x))^2*(1 - 0.5*(3*y))^3*(1.0*(3*x) - 1.0) - 3.51295090211393*(1 - 0.5*(3*x))^2*(1 - 0.5*(3*y))^2*(1.0*(3*x) - 1.0)*(0.25*(3*y) + 0.5) - 1.22561406884481*(1 - 0.5*(3*x))^2*(1.0*(3*x) - 1.0)*(0.5*(3*y) + 1)^3 - 0.00100089274721571*(1 - 0.5*(3*y))^3*(2.0 - 1.0*(3*x))*(1.0*(3*x) - 1.0)^2 - 0.0192041118681437*(1 - 0.5*(3*y))^3*(1.0*(3*x) - 1.0)^3 - 1.0284199907425*(1 - 0.5*(3*y))^2*(2.0 - 1.0*(3*x))*(1.0*(3*x) - 1.0)^2*(0.25*(3*y) + 0.5) - 0.38295897095889*(1 - 0.5*(3*y))^2*(1.0*(3*x) - 1.0)^3*(0.25*(3*y) + 0.5) - 0.319943178077135*(2.0 - 1.0*(3*x))*(1.0*(3*x) - 1.0)^2*(0.5*(3*y) + 1)^3 - 0.110331304802397*(1.0*(3*x) - 1.0)^3*(0.5*(3*y) + 1)^3)) + d)/3];
g_d=d^2;
% whole state space is [-2/3, 2/3]*[-2/3, 2/3], u(x) is constrained to ball 1 -(x^2 + y^2)>=0
H = 1 - x^2 - y^2;

% system constraints [-2/3, 2/3]*[-2/3, 2/3]
C0_0 = x^2 - 4/9;
C0_1 = y^2 - 4/9;

% f1 on constraints [-2/3, -1/3]*[-2/3, 2/3]
X1_0 = (x + 0.5)^2 - 1/36;
X1_1 = y^2 - 4/9;

% f2 on constraints [-1/3, 1/3]*[-2/3, 2/3]
X2_0 = x^2 - 1/9;
X2_1 = y^2 - 4/9;

% f3 on constraints [1/3, 2/3]*[-2/3, 2/3]
X3_0 = (x - 0.5)^2 - 1/36; 
X3_1 = y^2 - 4/9;

degree = 10;
x1=[x y];
x2 = [x y d];

[V,coe]= polynomial(x1,degree);

% h(x) = 1-x^2-y^2
% d=8
% obj=(pi*(1920*coe(1) + 480*coe(4) + 480*coe(6) + 240*coe(11) + 80*coe(13) + 240*coe(15) + 150*coe(22) + 30*coe(24) + 30*coe(26) + 150*coe(28) + 105*coe(37) + 15*coe(39) + 9*coe(41) + 15*coe(43) + 105*coe(45)))/1920;
% d = 10
obj=(pi*(7680*coe(1) + 1920*coe(4) + 1920*coe(6) + 960*coe(11) + 320*coe(13) + 960*coe(15) + 600*coe(22) + 120*coe(24) + 120*coe(26) + 600*coe(28) + 420*coe(37) + 60*coe(39) + 36*coe(41) + 60*coe(43) + 420*coe(45) + 315*coe(56) + 35*coe(58) + 15*coe(60) + 15*coe(62) + 35*coe(64) + 315*coe(66)))/7680;

V1=replace(V,x1,f1');
V2=replace(V,x1,f2');
V3=replace(V,x1,f3');
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

[s12,coe12]=polynomial(x1,degree1);
[s13,coe13]=polynomial(x1,degree1);

F=[sos(V-V1+s0*X1_0+s1*X1_1+s2*(g_d-0.05^2)-s3*H), sos(V-V2+s4*X2_0+s5*X2_1+s6*(g_d-0.05^2)-s7*H), sos(V-V3+s8*X3_0+s9*X3_1+s10*(g_d-0.05^2)-s11*H),
    sos((1+(C0_0)^2)*V-C0_0-s12*H), sos((1+(C0_1)^2)*V-C0_1-s13*H), sos(s0),sos(s1), sos(s2),sos(s3),sos(s4),sos(s5),sos(s6),sos(s7),sos(s8),sos(s9),sos(s10),sos(s11),sos(s12),sos(s13)];

ops = sdpsettings('solver','mosek','sos.newton',1,'sos.congruence',1);
diagnostics=solvesdp(F,obj,ops,[coe;coe0;coe1;coe2;coe3;coe4;coe5;coe6;coe7;coe8;coe9;coe10;coe11;coe12;coe13]);

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
%d=10? disturbance=0.2 scale
%-0.0186297374616+0.000326042536362*x+7.87795721704e-05*y-0.0245403645255*x^2-0.0145569110953*x^2*y-0.059774607903*x^3+0.000778671180061*x*y+0.737707316033*x^3*y-0.000508694408648*y^2-0.00962785698989*x*y^2-0.00224906344622*y^3+0.514446215841*x^2*y^2+0.238589081278*x^2*y^3+0.343583513402*x^3*y^2+0.000509332189018*x*y^3+3.49071967837*x^3*y^3+2.54848867822*x^4+0.00742730412341*y^4+1.04870165096*x^5+0.141134378896*x^4*y+0.14439498584*x*y^4+0.0285877824801*y^5-5.83768040184*x^6-2.51641802335*x^5*y+3.01569530345*x^4*y^2+1.73654696876*x^2*y^4+1.21217614154*x*y^5-0.0639716949012*y^6-2.25465038144*x^7-0.40332413766*x^6*y-1.44972371159*x^5*y^2-0.220769222908*x^4*y^3+0.64105658932*x^3*y^4-0.577750929415*x^2*y^5-0.726679077695*x*y^6-0.0900749354262*y^7+5.74618257194*x^8+3.92108620987*x^7*y-16.3355156347*x^6*y^2-11.9611447227*x^5*y^3-2.38649911655*x^4*y^4-4.36505302442*x^3*y^5-4.88659944522*x^2*y^6-2.50705929521*x*y^7+1.50145382304*y^8+1.29930633227*x^9+0.320820263513*x^8*y+0.952381536266*x^7*y^2+0.0361037148917*x^6*y^3+0.869677706192*x^5*y^4+0.10685560055*x^4*y^5-1.84875451629*x^3*y^6+0.397316171519*x^2*y^7+0.863141576896*x*y^8+0.0771594074445*y^9-1.93438405021*x^10-2.38824191419*x^9*y+14.6017559257*x^8*y^2+8.53939158801*x^7*y^3+3.37487447424*x^6*y^4+5.40811901492*x^5*y^5+4.55612395012*x^4*y^6+4.38009060668*x^3*y^7-1.01656714635*x^2*y^8+0.0129416741164*x*y^9
%d=10, disturbance=0.15 scale
%-0.174587358925+0.000858693867599*x+5.66680252946e-05*y-0.0352895495475*x^2+0.0220438120836*x^2*y-0.112581728789*x^3+0.00733961494636*x*y+1.82128187084*x^3*y+0.00465590010632*y^2-0.0160741210207*x*y^2-0.00779210472182*y^3+2.42300274105*x^2*y^2-0.323178911183*x^2*y^3+0.794326766824*x^3*y^2+0.331170819124*x*y^3-5.23228671712*x^3*y^3+9.99688671891*x^4-0.209562407226*y^4+1.8968351846*x^5-0.752090358099*x^4*y+0.316967834573*x*y^4+0.182406077306*y^5-36.5344102839*x^6-5.88508357506*x^5*y-8.40192405614*x^4*y^2-9.96854786046*x^2*y^4-1.03194760443*x*y^5+3.82867276191*y^6-4.25316817803*x^7+2.14940294955*x^6*y-4.18880659461*x^5*y^2+0.667784384553*x^4*y^3+0.158243413391*x^3*y^4+2.10487622291*x^2*y^5-1.05151859083*x*y^6-0.571332212868*y^7+49.0487571986*x^8+6.19120835233*x^7*y+13.296642833*x^6*y^2+7.87899484226*x^5*y^3+15.3555880843*x^4*y^4+16.8850959737*x^3*y^5+7.35538571231*x^2*y^6+0.0547573370134*x*y^7-2.98361601727*y^8+2.47037574457*x^9-1.46960937033*x^8*y+4.29330275282*x^7*y^2-1.19085459687*x^6*y^3+1.23827600482*x^5*y^4-1.40675214558*x^4*y^5-1.01329438232*x^3*y^6-1.57660554269*x^2*y^7+0.898952062579*x*y^8+0.374381699674*y^9-21.872313823*x^10-2.18146816384*x^9*y-10.0767112516*x^8*y^2-2.40121118614*x^7*y^3+1.54264129369*x^6*y^4-16.6042112118*x^5*y^5-15.0005332471*x^4*y^6-10.5692458164*x^3*y^7-0.298518189107*x^2*y^8+0.232189000965*x*y^9
%d=8, disturbance=0.15, scale
%-0.095436563201-0.000382159771392*x+1.68086216873e-05*y-0.01451605853*x^2+0.00378187246423*x^2*y-0.0555735370445*x^3+0.00384693270147*x*y+0.781348680033*x^3*y+0.002653160795*y^2-0.00507878207996*x*y^2-0.00381024830072*y^3+1.00010295462*x^2*y^2+0.107000373828*x^2*y^3+0.458673374498*x^3*y^2+0.066107269952*x*y^3-2.21568660734*x^3*y^3+4.34672831351*x^4-0.117279771147*y^4+0.606579131776*x^5-0.371187783953*x^4*y+0.0441326391988*x*y^4+0.0772325295543*y^5-9.43971870543*x^6-1.10020195889*x^5*y-2.60514389836*x^4*y^2-2.65497010858*x^2*y^4+1.27602975*x*y^5+1.90155265004*y^6-0.71324610614*x^7+0.587265305521*x^6*y-0.623780060208*x^5*y^2-0.361742490539*x^4*y^3-0.435023647483*x^3*y^4+0.354125204405*x^2*y^5-0.0190864840329*x*y^6-0.157833151488*y^7+6.11133421911*x^8+0.189943540652*x^7*y+2.15680712167*x^6*y^2+0.626772370452*x^5*y^3+1.94699435254*x^4*y^4+3.98833488191*x^3*y^5+0.485385633733*x^2*y^6-2.67713762155*x*y^7-0.821606028679*y^8
%d=8, disturbance=0.1, scale
%-0.185158249313-0.000473119064748*x+7.140793824e-05*y+0.00836356573883*x^2+0.0253138051056*x^2*y+0.309155891661*x^3+0.00465698950305*x*y+0.571994549891*x^3*y+0.00413435632066*y^2+0.0394295044232*x*y^2-0.000372159449832*y^3+1.30769403886*x^2*y^2+0.111360475954*x^2*y^3+0.112562431688*x^3*y^2+0.344773262005*x*y^3-0.296721683604*x^3*y^3+5.49744084087*x^4-0.15799424306*y^4-0.766154330592*x^5-0.119272033703*x^4*y-0.25461634796*x*y^4-0.0141542652043*y^5-12.0471160801*x^6-1.1203197395*x^5*y-2.09261930534*x^4*y^2-6.88983225391*x^2*y^4-1.31921472181*x*y^5+4.00683818715*y^6+0.46954453433*x^7+0.131881504241*x^6*y-0.114360742575*x^5*y^2-0.183219552511*x^4*y^3-0.0801923298815*x^3*y^4+0.0158830858269*x^2*y^5+0.209515768697*x*y^6+0.00262033253548*y^7+7.74893172914*x^8+0.419448892336*x^7*y+1.27519142123*x^6*y^2-0.113501170791*x^5*y^3+4.63195071404*x^4*y^4+1.24318178509*x^3*y^5+3.46588491504*x^2*y^6+0.842679004196*x*y^7-3.20186643367*y^8
%d=8, disturbance=0.05
%-0.202365181636-0.000215009023518*x-1.70396414056e-05*y+0.0579883929434*x^2+0.0218489517004*x^2*y+0.312256054801*x^3+0.00667207216021*x*y+0.379472029987*x^3*y+0.00545236408164*y^2+0.0473604175964*x*y^2+0.00775023891412*y^3+1.38079717409*x^2*y^2+0.0913008695767*x^2*y^3-0.0191234259779*x^3*y^2+0.847668246321*x*y^3-0.573961210864*x^3*y^3+3.39494195498*x^4-0.130120070322*y^4-0.862579484751*x^5-0.0621650388833*x^4*y-0.311001976429*x*y^4-0.0492159781768*y^5-6.31541418556*x^6-0.709902456876*x^5*y-2.55473582039*x^4*y^2-6.71979264816*x^2*y^4-3.05080936341*x*y^5+4.30896302621*y^6+0.597053864936*x^7+0.0232154822223*x^6*y+0.0168917541206*x^5*y^2-0.032525497937*x^4*y^3+0.101338803745*x^3*y^4-0.0729773972423*x^2*y^5+0.243348816782*x*y^6+0.0438214864079*y^7+3.77676086304*x^8+0.157443976319*x^7*y+1.67738304589*x^6*y^2-0.0562328419816*x^5*y^3+4.58131723787*x^4*y^4+1.74615514317*x^3*y^5+3.36050391139*x^2*y^6+2.1615856001*x*y^7-3.55293798867*y^8
%d=10, disturbance=0.05
%-0.263913574764-0.000528242941314*x+0.000223494053431*y+0.136880834381*x^2+0.0453223772422*x^2*y+0.451060956722*x^3+0.0286858606342*x*y+0.149573897799*x^3*y+0.00286877418857*y^2+0.0756223492812*x*y^2-0.0604702390379*y^3+2.14093826624*x^2*y^2-0.201502301296*x^2*y^3-0.275534348368*x^3*y^2+1.86352665373*x*y^3-8.18848993039*x^3*y^3+7.31224043295*x^4+1.41684160955*y^4-1.83528241165*x^5-0.257056130739*x^4*y-0.543601992228*x*y^4+0.392191892045*y^5-26.1600398701*x^6+1.28595570938*x^5*y-9.58232354246*x^4*y^2-17.6272057479*x^2*y^4-6.71171067242*x*y^5+0.711968481727*y^6+2.44444490116*x^7+0.492679440713*x^6*y+0.552677199957*x^5*y^2+0.549808945833*x^4*y^3+0.671044748426*x^3*y^4+0.857410587542*x^2*y^5+0.750937834496*x*y^6-0.828533309032*y^7+35.5610260815*x^8-4.47956327485*x^7*y+20.8953724365*x^6*y^2+6.83780968103*x^5*y^3+20.431214508*x^4*y^4+30.3080551192*x^3*y^5+26.3674502935*x^2*y^6+4.46487516982*x*y^7-1.44282491053*y^8-1.05931697212*x^9-0.285142290228*x^8*y-0.510256189167*x^7*y^2-0.366656219615*x^6*y^3+0.31058970607*x^5*y^4-1.28400293855*x^4*y^5-0.80943643865*x^3*y^6-0.421014759443*x^2*y^7-0.271029456555*x*y^8+0.496748239549*y^9-16.1608007815*x^10+3.00943525546*x^9*y-16.0941143617*x^8*y^2+1.09055359424*x^7*y^3-4.56595616916*x^6*y^4-22.8329695109*x^5*y^5-19.4142461099*x^4*y^6-18.6447698705*x^3*y^7-11.6454705747*x^2*y^8+0.364939418082*x*y^9