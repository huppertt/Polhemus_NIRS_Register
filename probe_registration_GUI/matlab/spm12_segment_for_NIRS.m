function outfile = spm12_segment_for_NIRS(t1file, outfile, spmdir, lesionfile)
% SPM12_SEGMENT_FOR_NIRS  Segment a T1w MRI into a single NIRS tissue-label volume.
%
%   outfile = spm12_segment_for_NIRS('T1w.nii')
%   outfile = spm12_segment_for_NIRS('T1w.nii', 'segmented.nii.gz')
%   outfile = spm12_segment_for_NIRS('T1w.nii', [], '/path/to/spm12')
%   outfile = spm12_segment_for_NIRS('T1w.nii', [], [], 'lesion.nii.gz')
%
% Runs SPM12 unified segmentation (6 tissue classes) and combines the native
% space probability maps into one uint8 label volume on the T1 voxel grid:
%
%     0 = air / background      3 = CSF
%     1 = skin (scalp)          4 = gray matter
%     2 = skull                 5 = white matter
%                               6 = lesion (only if a lesion mask is given)
%
% Each voxel takes the most probable class.  Disconnected specks outside
% the head are removed (largest connected head component is kept).
%
% Inputs
%   t1file  : T1-weighted image (.nii or .nii.gz)
%   outfile : output file (default: <T1 folder>/segmented.nii.gz)
%   spmdir  : SPM12 folder (default: ~/spm12 if present, otherwise
%             ~/Documents/GitHub/nirs-toolbox-extra/spm12)
%   lesionfile : optional manually drawn lesion mask (.nii or .nii.gz) in
%             the same scanner space as the T1 (any voxel grid; it is
%             resampled onto the T1 grid).  Lesion voxels labelled CSF, gray
%             or white matter become label 6; lesion voxels in air, skin or
%             skull are left unchanged (reported, as they usually indicate a
%             misplaced mask).  Default output is then segmented_lesion.nii.gz.
%
% SPM writes its intermediate files to a temporary folder that is deleted
% afterwards, so the subject folder only receives the final label file.

if nargin < 3 || isempty(spmdir)
    % Prefer a local copy: the GitHub/nirs-toolbox-extra copy lives in the
    % iCloud-synced Documents folder and its files may be cloud-only
    % placeholders, which makes SPM extremely slow (each file is fetched on demand).
    candidates = {fullfile(char(java.lang.System.getProperty('user.home')), 'spm12'), ...
        fullfile(char(java.lang.System.getProperty('user.home')), ...
        'Documents', 'GitHub', 'nirs-toolbox-extra', 'spm12')};
    spmdir = '';
    for c = 1:numel(candidates)
        if isfile(fullfile(candidates{c}, 'spm.m'))
            spmdir = candidates{c};
            break
        end
    end
    if isempty(spmdir)
        error('SPM12 not found; pass its folder as the third argument.');
    end
end
% Put SPM first on the path so that copies of SPM functions shipped with
% other toolboxes (e.g. fieldtrip/external) cannot shadow it.
addpath(spmdir);
if ~strcmp(fileparts(which('spm')), spmdir)
    error('spm.m resolves to %s instead of %s', which('spm'), spmdir);
end
if ~strcmp(spm('Ver'), 'SPM12')
    error('SPM12 is required (found %s).', spm('Ver'));
end
fprintf('Using SPM12 from %s\n', spmdir);

t1file = char(t1file);
if ~isfile(t1file)
    error('File not found: %s', t1file);
end
[t1dir, ~, ~] = fileparts(which_or_abs(t1file));
if nargin < 4
    lesionfile = '';
end
lesionfile = char(lesionfile);
if ~isempty(lesionfile) && ~isfile(lesionfile)
    error('Lesion file not found: %s', lesionfile);
end
if nargin < 2 || isempty(outfile)
    if isempty(lesionfile)
        outfile = fullfile(t1dir, 'segmented.nii.gz');
    else
        outfile = fullfile(t1dir, 'segmented_lesion.nii.gz');
    end
end
outfile = char(outfile);

% --- Work on a copy in a temp folder so SPM's c*/seg8 files don't clutter the subject folder
work = tempname;
mkdir(work);
cleanupObj = onCleanup(@() rmdir(work, 's'));
if endsWith(lower(t1file), '.gz')
    f = gunzip(t1file, work);
    t1work = f{1};
else
    t1work = fullfile(work, 'T1w.nii');
    copyfile(t1file, t1work);
end

% --- SPM12 unified segmentation (defaults of the SPM12 Segment batch)
spm('defaults', 'fmri');
spm_jobman('initcfg');
tpm = fullfile(spm('Dir'), 'tpm', 'TPM.nii');
ngaus = [1 1 2 3 4 2];
job = struct;
job.channel.vols = {[t1work ',1']};
job.channel.biasreg = 0.001;
job.channel.biasfwhm = 60;
job.channel.write = [0 0];
for k = 1:6
    job.tissue(k).tpm = {sprintf('%s,%d', tpm, k)};
    job.tissue(k).ngaus = ngaus(k);
    job.tissue(k).native = [1 0];      % native-space probability maps c1..c6
    job.tissue(k).warped = [0 0];
end
job.warp.mrf = 1;
job.warp.cleanup = 1;
job.warp.reg = [0 0.001 0.5 0.05 0.2];
job.warp.affreg = 'mni';
job.warp.fwhm = 0;
job.warp.samp = 3;
job.warp.write = [0 0];
matlabbatch{1}.spm.spatial.preproc = job;
fprintf('Running SPM12 segmentation on %s ...\n', t1file);
spm_jobman('run', matlabbatch);

% --- Combine probability maps into labels
[~, nm] = fileparts(t1work);
V = spm_vol(fullfile(work, sprintf('c1%s.nii', nm)));
P = zeros([V.dim 6], 'single');
for k = 1:6
    P(:, :, :, k) = spm_read_vols(spm_vol(fullfile(work, sprintf('c%d%s.nii', k, nm))));
end
% SPM classes: 1 GM, 2 WM, 3 CSF, 4 bone, 5 soft tissue, 6 air  ->  NIRS labels
spm2nirs = uint8([4 5 3 2 1 0]);
[pmax, idx] = max(P, [], 4);
lab = spm2nirs(idx);
lab(pmax <= 0) = 0;

% Keep the largest connected head component (removes noise/specks in the background)
[cc, n] = spm_bwlabel(double(lab > 0), 26);
if n > 1
    sz = accumarray(cc(cc > 0), 1);
    [~, keep] = max(sz);
    lab(cc ~= keep) = 0;
end

Vin = spm_vol(t1work);

% --- Optional lesion mask -> label 6 (inside CSF / GM / WM only)
if ~isempty(lesionfile)
    les = lesion_on_grid(lesionfile, Vin(1), work);
    inbrain = les & ismember(lab, uint8([3 4 5]));
    nout = nnz(les) - nnz(inbrain);
    lab(inbrain) = 6;
    fprintf('Lesion: %d voxels set to label 6', nnz(inbrain));
    if nout > 0
        fprintf(' (%d lesion voxels in air/skin/skull left unchanged - check the mask is in T1 space)', nout);
    end
    fprintf('\n');
end

% --- Write uint8 label image on the T1 grid
Vo = Vin(1);
[odir, oname, oext] = fileparts(outfile);
if isempty(odir), odir = pwd; end
if strcmpi(oext, '.gz')
    [~, oname] = fileparts(oname);   % strip .nii
    gz = true;
else
    gz = false;
end
Vo.fname = fullfile(odir, [oname '.nii']);
Vo.dt = [spm_type('uint8') spm_platform('bigend')];
Vo.pinfo = [1; 0; 0];
Vo.n = [1 1];
Vo.descrip = 'NIRS seg 0air 1skin 2skull 3CSF 4GM 5WM 6lesion (SPM12)';
Vo = rmfield_if(Vo, 'private');
spm_write_vol(Vo, double(lab));
if gz
    gzip(Vo.fname);
    delete(Vo.fname);
    outfile = [Vo.fname '.gz'];
else
    outfile = Vo.fname;
end

names = {'air', 'skin', 'skull', 'CSF', 'gray', 'white', 'lesion'};
vox = abs(det(Vin(1).mat(1:3, 1:3)));
fprintf('Wrote %s\n', outfile);
for k = 0:5 + any(lab(:) == 6)
    fprintf('  %d %-6s %8.1f cm^3\n', k, names{k + 1}, nnz(lab == k) * vox / 1000);
end
end

function les = lesion_on_grid(lesionfile, Vref, work)
% Binary lesion mask resampled (via the scanner-space affines) onto the grid of Vref.
if endsWith(lower(lesionfile), '.gz')
    f = gunzip(lesionfile, fullfile(work, 'lesion'));
    lesionfile = f{1};
end
Vl = spm_vol(lesionfile);
Vl = Vl(1);
Y = spm_read_vols(Vl);
Y(~isfinite(Y)) = 0;
if all(Y(:) == round(Y(:)))
    B = double(Y > 0);                         % 0/1, 0/255 or several lesion labels
else
    B = double(Y >= 0.5 * min(1, max(Y(:))));  % fractional / probabilistic mask
end
if isequal(Vl.dim(1:3), Vref.dim(1:3)) && max(abs(Vl.mat(:) - Vref.mat(:))) < 1e-3
    les = B > 0;
    return
end
[i, j, k] = ndgrid(1:Vref.dim(1), 1:Vref.dim(2), 1:Vref.dim(3));
M = Vl.mat \ Vref.mat;                        % reference voxel -> lesion voxel (1-based)
x = M(1,1)*i + M(1,2)*j + M(1,3)*k + M(1,4);
y = M(2,1)*i + M(2,2)*j + M(2,3)*k + M(2,4);
z = M(3,1)*i + M(3,2)*j + M(3,3)*k + M(3,4);
les = interpn(B, x, y, z, 'linear', 0) >= 0.5;
end

function p = which_or_abs(f)
if java.io.File(f).isAbsolute()
    p = f;
else
    p = fullfile(pwd, f);
end
end

function s = rmfield_if(s, f)
if isfield(s, f)
    s = rmfield(s, f);
end
end
