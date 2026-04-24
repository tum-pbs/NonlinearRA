%% Jacobian for lorenz system
function J = jacobian_ana(b,sig,r,x0)

    J = [
        [   -sig,   sig,      0];
        [r-x0(3),    -1, -x0(1)];
        [  x0(2), x0(1),     -b];
    ];

end
