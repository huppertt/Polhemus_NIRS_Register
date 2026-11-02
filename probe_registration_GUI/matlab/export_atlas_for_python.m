function export_atlas_for_python(sessionfile, probefile, outfile)
% Export atlas/template arrays, not subject measurements, for the Python GUI.
% Requires the NIRS-toolbox on the MATLAB path.
% export_atlas_for_python('/path/to/session.mat', '/path/to/sample_probeInfo.mat')
if nargin < 3 || isempty(outfile)
    outfile = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
        'assets', 'atlas_ICBM152_and_probe.mat');
end
d=load(sessionfile, 'Atlas'); A=d.Atlas;
s=struct;
for k=1:numel(A)
    s.(sprintf('nodes%d',k))=A(k).nodes;
    s.(sprintf('faces%d',k))=A(k).faces;
end
fid=A(1).fiducials;
s.fid_name=fid.Name; s.fid_xyz=[fid.X fid.Y fid.Z];
po=nirs.io.loadNIRxProbe(probefile);
s.opt_name=po.optodes_registered.Name; s.opt_type=po.optodes_registered.Type;
s.opt_xyz=[po.optodes_registered.X po.optodes_registered.Y po.optodes_registered.Z];
s.opt2d=[po.optodes.X po.optodes.Y];
s.link=[po.link.source po.link.detector];
save(outfile,'-struct','s','-v7');
end
