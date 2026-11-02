function export_tissue_optics(outfile, toolboxdir)
% EXPORT_TISSUE_OPTICS  Tabulate nirs.media.tissues optical properties for Python.
%
%   export_tissue_optics                      % writes ../assets/nirs_tissue_optics.json
%   export_tissue_optics(outfile, toolboxdir)
%
% Writes the NIRS-toolbox tissue optical properties on a 1-nm grid so the
% Python MCX export (mcx_export.py) uses exactly the same values:
%   skin, bone, water (CSF)  : nirs.media.tissues.skin / bone / water
%   brain spectra            : HbO, HbR, water and fat absorption from
%                              nirs.media.getspectra plus the SpectralProp
%                              defaults (a, b, g, ri, water, lipid), so brain
%                              mua/mus can be computed for any SO2 / HbT.
% Units: mua, mus in 1/mm; HbO/HbR extinction such that
%   mua = ext_hbo*HbO[uM]*1e-6 + ext_hbr*HbR[uM]*1e-6 + water*mua_water + lipid*mua_fat

if nargin < 1 || isempty(outfile)
    outfile = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
        'assets', 'nirs_tissue_optics.json');
end
if nargin < 2 || isempty(toolboxdir)
    toolboxdir = '';
end
if isempty(which('nirs.media.tissues.skin'))
    if isempty(toolboxdir)
        error('Add the NIRS-toolbox to the MATLAB path or pass toolboxdir.');
    end
    addpath(toolboxdir);
end

lambda = 650:950;
out = struct();
out.source = 'nirs-toolbox nirs.media.tissues / nirs.media.getspectra';
out.created = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm'));
out.lambda_nm = lambda;

for name = {'skin', 'bone', 'water'}
    p = feval(['nirs.media.tissues.' name{1}], lambda);
    t = struct('mua', p.mua, 'mus', p.mus, 'g', p.g(1), 'ri', p.ri(1));
    t.mua(~isfinite(t.mua)) = NaN;
    t.mus(~isfinite(t.mus)) = NaN;
    out.(name{1}) = t;
end

ext = nirs.media.getspectra(lambda(:));
sp = nirs.media.SpectralProp();             % class defaults
out.spectra = struct('ext_hbo', ext(:, 1)', 'ext_hbr', ext(:, 2)', 'mua_water', ext(:, 3)', ...
    'mua_fat', ext(:, 4)');
out.brain_defaults = struct('a', sp.a, 'b', sp.b, 'g', sp.g(1), 'ri', sp.ri, 'water', sp.water, ...
    'lipid', sp.lipid, 'so2', 0.7, 'hbt', 50, 'lambda_min', 650, 'lambda_max', 900);

% reference values for checking the Python implementation
chk = [690 780 830 850];
b = nirs.media.tissues.brain(chk, 0.7, 50);
out.check = struct('lambda', chk, 'brain_mua', b.mua, 'brain_mus', b.mus);

txt = jsonencode(out);
txt = strrep(txt, 'null', 'NaN');           % keep NaN (outside a table's range) readable by Python json
fid = fopen(outfile, 'w');
fwrite(fid, txt);
fclose(fid);
fprintf('Wrote %s\n', outfile);
end
