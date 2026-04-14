function mni_mm = map_native_to_mni(subj_folder, native_mm, transform_folder_name)
%MAP_NATIVE_TO_MNI Map native-space points to MNI using Lead-DBS transforms.
%
% mni_mm = map_native_to_mni(subj_folder, native_mm)
% mni_mm = map_native_to_mni(subj_folder, native_mm, transform_folder_name)
%
% Inputs
% ------
% subj_folder : Lead-DBS subject folder, e.g. .../derivatives/leaddbs/sub-001
% native_mm   : [N x 3] native coordinates in mm
% transform_folder_name : deformation folder under subj_folder (default: 'inverseTransform')
%
% Output
% -------
% mni_mm      : [N x 3] mapped coordinates in MNI mm

arguments
    subj_folder (1,:) char
    native_mm (:,3) double
    transform_folder_name (1,:) char = 'inverseTransform'
end

if isempty(native_mm)
    mni_mm = zeros(0,3);
    return;
end

require_functions({"ea_getptopts", "ea_load_nii", "ea_map_coords"});

opt = ea_getptopts(subj_folder);

if ~isfield(opt, 'subj') || ~isfield(opt.subj, 'coreg') || ...
        ~isfield(opt.subj.coreg, 'anat') || ~isfield(opt.subj.coreg.anat, 'preop')
    error('Lead-DBS options missing opt.subj.coreg.anat.preop.');
end
if ~isfield(opt.subj, 'AnchorModality')
    error('Lead-DBS options missing opt.subj.AnchorModality.');
end
if ~isfield(opt.subj, 'subjDir')
    error('Lead-DBS options missing opt.subj.subjDir.');
end

anchor = opt.subj.AnchorModality;
if ~isfield(opt.subj.coreg.anat.preop, anchor)
    error('Anchor modality "%s" not found in opt.subj.coreg.anat.preop.', anchor);
end

anat_nii_path = opt.subj.coreg.anat.preop.(anchor);
nii = ea_load_nii(anat_nii_path);
if ~isfield(nii, 'mat') || ~isfield(nii, 'fname')
    error('ea_load_nii output missing required fields "mat" and/or "fname".');
end

scrf_mm = native_mm;

scrf_affine_path = '';
if isfield(opt.subj, 'brainshift') && isfield(opt.subj.brainshift, 'transform') && ...
        isfield(opt.subj.brainshift.transform, 'scrf')
    scrf_affine_path = opt.subj.brainshift.transform.scrf;
end
if ~isempty(scrf_affine_path) && isfile(scrf_affine_path)
    d = load(scrf_affine_path);
    if ~isfield(d, 'mat')
        error('SCRF affine file missing variable "mat": %s', scrf_affine_path);
    end
    A = double(d.mat);
    if ~isequal(size(A), [4 4])
        error('SCRF affine matrix must be 4x4: %s', scrf_affine_path);
    end
    n = size(scrf_mm, 1);
    tmp = (A * [scrf_mm, ones(n,1)]')';
    scrf_mm = tmp(:,1:3);
end

transform_folder = fullfile(opt.subj.subjDir, transform_folder_name);
n = size(scrf_mm, 1);
vox = (nii.mat \ [scrf_mm, ones(n,1)]')';
vox = vox(:,1:3);

% ea_map_coords expects [3 x N]
def = ea_map_coords(vox', nii.fname, transform_folder, '');
mni_mm = def';

if size(mni_mm, 2) ~= 3
    error('Mapped coordinates must be Nx3, got size: [%d %d].', size(mni_mm,1), size(mni_mm,2));
end

end
