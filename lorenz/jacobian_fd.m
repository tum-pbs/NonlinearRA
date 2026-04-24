function J = jacobian_fd(net, x0, h)

% This function computes a numerical Jacobian for a feedforward neural
% network using central finite differences. Since MATLAB's feedforwardnet
% does not support automatic differentiation, we approximate partial
% derivatives by perturbing each input dimension independently.
% The function takes:
%   net — a trained feedforwardnet object (3→3 mapping in this project)
%   x0  — a 3×1 input state at which the Jacobian is evaluated
%   h   — finite-difference step size (default: 1e-6)
% The output is:
%   J — a 3×3 Jacobian matrix whose columns correspond to partial
%        derivatives with respect to x, y, z.

if nargin<3, h = 1e-6; end
    y0 = net(x0);           
    J  = zeros(3,3);
    for j = 1:3
        dx    = zeros(3,1);
        dx(j) = h;
        y_plus  = net(x0 + dx);
        y_minus = net(x0 - dx);
        J(:,j)= (y_plus - y_minus) / 2 / h;
    end
end