function coords = contact_midpoint_coords(subj_folder, options)
%CONTACT_MIDPOINT_COORDS Compute midpoints between adjacent contacts in native/SCRF/MNI.
%
% This function reads reconstructed contact coordinates from Lead-DBS
% (reco.native.coords_mm) and returns the midpoints between adjacent
% contacts, optionally mapped to SCRF and MNI space.
%
% Inputs
% ------
% subj_folder
%   Subject folder managed by Lead-DBS (e.g., .../leaddbs/sub-001).
%
% Name-Value Options
% ------------------
% SavePath (char)             : Output MAT file path. If empty, do not save.
%                               If only a file name is provided, it is saved
%                               in subj_folder. (default: 'contact_midpoint_coords.mat')
% Overwrite (logical)         : If false and output exists, skip computation and
%                               return the saved coords. (default: true)
% MarkerSpace (char)          : Must be 'native' (kept for API stability; no other spaces supported).
% TransformFolderName (char)  : Folder under opt.subj.subjDir containing deformation fields
%                               for ea_map_coords (default: 'inverseTransform').
% Verbose (logical)           : Print progress messages (default: true).
%
% Output
% ------
% coords (struct)
%   .native     {side} (N-1)x3 midpoints in native space (mm)
%   .scrf       {side} (N-1)x3 midpoints in SCRF space (mm)
%   .mni        {side} (N-1)x3 midpoints in MNI space (mm)
%   .contacts_native {side} Nx3 original contact centers in native space (mm)
%   .axis       struct with head/tail/dir_unit in native space (if markers available)
%   .meta       struct with provenance and file paths
%
% Dependencies
% ------------
% Requires Lead-DBS on MATLAB path:
%   ea_getptopts, ea_load_nii, ea_map_coords
%
% Notes
% -----
% - Midpoints are computed as the mean of adjacent contacts in native space.
% - All coordinates are assumed to be in millimeters.
%

arguments
    subj_folder (1,:) char
    options.SavePath (1,:) char = 'contact_midpoint_coords.mat'
    options.Overwrite (1,1) logical = true
    options.MarkerSpace (1,:) char = 'native'
    options.TransformFolderName (1,:) char = 'inverseTransform'
    options.Verbose (1,1) logical = true
end

% Resolve save path early (used for overwrite checks).
save_path = resolve_save_path(options.SavePath, subj_folder);
if ~options.Overwrite && ~isempty(save_path) && isfile(save_path)
    if options.Verbose
        fprintf('[%s] contact_midpoint_coords: output exists; skipping (Overwrite=false): %s\n', ...
            string(datetime()), save_path);
    end
    s = load(save_path, 'coords');
    if ~isfield(s, 'coords')
        error('Existing file does not contain variable "coords": %s', save_path);
    end
    coords = s.coords;
    return;
end

require_functions({"ea_getptopts", "ea_load_nii", "ea_map_coords"});

if options.Verbose
    fprintf('[%s] contact_midpoint_coords: loading Lead-DBS options...\n', string(datetime()));
end
opt = ea_getptopts(subj_folder);

sides = opt.sides;

% Load Lead-DBS reconstruction (contact centers).
[reco, reco_path] = load_reco(opt);

if ~strcmpi(options.MarkerSpace, 'native')
    error('Only MarkerSpace="native" is supported. Got: %s', options.MarkerSpace);
end

marker_space = 'native';
validate_native_coords(reco);

% Compute native midpoint coordinates.
[native_cell, axis_info, contact_cell] = contacts_to_midpoint_coords(reco, sides);

% Apply SCRF affine (brainshift) if available.
[scrf_cell, scrf_affine_path] = apply_scrf_affine_if_present(native_cell, sides, opt);

% Warp SCRF -> MNI (nonlinear deformation fields).
[mni_cell, anat_nii_path, transform_folder] = warp_scrf_to_mni(scrf_cell, sides, opt, options.TransformFolderName);

% Assemble output.
coords = struct();
coords.native = native_cell;
coords.scrf = scrf_cell;
coords.mni = mni_cell;
coords.contacts_native = contact_cell;
coords.axis = axis_info;
coords.meta = struct();
coords.meta.created_at = string(datetime());
coords.meta.units = "mm";
coords.meta.subj_folder = subj_folder;
coords.meta.reco_path = reco_path;
coords.meta.marker_space = marker_space;
coords.meta.scrf_affine_path = scrf_affine_path;
coords.meta.anat_nii_path = anat_nii_path;
coords.meta.transform_folder = transform_folder;
coords.meta.leaddbs_sides = sides;
coords.meta.native_axis = "RAS+ (Lead-DBS native space)";
coords.meta.midpoint_definition = "mean of adjacent contact centers in native space";

% Optional save.
if ~isempty(save_path)
    if options.Verbose
        fprintf('[%s] contact_midpoint_coords: saving result...\n', string(datetime()));
    end
    save(save_path, 'coords');
end

if options.Verbose
    fprintf('[%s] contact_midpoint_coords: done.\n', string(datetime()));
end

end

% ========================================================================
function save_path = resolve_save_path(save_path_in, subj_folder)
% Resolve save path; place bare filenames in subject folder.

save_path = save_path_in;
if isempty(save_path)
    return;
end

if isempty(fileparts(save_path))
    save_path = fullfile(subj_folder, save_path);
end

end

% ========================================================================
function [reco, reco_path] = load_reco(opt)
% Load Lead-DBS reconstruction file containing lead markers.

if ~isfield(opt, 'subj') || ~isfield(opt.subj, 'recon') || ~isfield(opt.subj.recon, 'recon')
    error('Lead-DBS options do not contain opt.subj.recon.recon.');
end

reco_path = opt.subj.recon.recon;
if ~isfile(reco_path)
    error('Lead-DBS reconstruction file not found: %s', reco_path);
end

s = load(reco_path, 'reco');
if ~isfield(s, 'reco')
    error('Reconstruction file does not contain variable "reco": %s', reco_path);
end
reco = s.reco;

end

% ========================================================================
function validate_native_coords(reco)
% Validate that reco.native.coords_mm exists.

if ~isfield(reco, 'native')
    error('reco is missing required field "native".');
end
if ~isfield(reco.native, 'coords_mm')
    error('reco.native is missing required field "coords_mm".');
end

end

% ========================================================================
function [mid_cell, axis_info, contact_cell] = contacts_to_midpoint_coords(reco, sides)
% Compute midpoints between adjacent contact centers in native space.

n_sides = max(sides);
mid_cell = cell(1, n_sides);
contact_cell = cell(1, n_sides);

axis_info = struct();
axis_info.head_native_mm = cell(1, n_sides);
axis_info.tail_native_mm = cell(1, n_sides);
axis_info.dir_unit_native = cell(1, n_sides);

for i = 1:n_sides
    mid_cell{i} = zeros(0,3);
    contact_cell{i} = zeros(0,3);
    axis_info.head_native_mm{i} = [NaN, NaN, NaN];
    axis_info.tail_native_mm{i} = [NaN, NaN, NaN];
    axis_info.dir_unit_native{i} = [NaN, NaN, NaN];
end

markers_available = isfield(reco, 'native') && isfield(reco.native, 'markers');

for side = sides
    coords_mm = get_contact_coords(reco, side);
    contact_cell{side} = coords_mm;

    n = size(coords_mm, 1);
    if n >= 2
        mid_cell{side} = (coords_mm(1:end-1, :) + coords_mm(2:end, :)) ./ 2;
    else
        mid_cell{side} = zeros(0, 3);
    end

    if markers_available
        [head_mm, tail_mm] = get_head_tail(reco, side);
        axis_vec = tail_mm - head_mm;
        axis_norm = norm(axis_vec);
        if axis_norm == 0
            error('Lead axis has zero length for side %d. Check markers in reconstruction.', side);
        end
        dir_unit = axis_vec ./ axis_norm; % head -> tail

        axis_info.head_native_mm{side} = head_mm;
        axis_info.tail_native_mm{side} = tail_mm;
        axis_info.dir_unit_native{side} = dir_unit;
    end
end

end

% ========================================================================
function coords_mm = get_contact_coords(reco, side)
% Retrieve contact center coordinates for a given side.

coords_mm = zeros(0, 3);

if ~isfield(reco, 'native') || ~isfield(reco.native, 'coords_mm')
    error('reco.native.coords_mm is missing.');
end

coords_cell = reco.native.coords_mm;
if numel(coords_cell) < side || isempty(coords_cell{side})
    return;
end

coords_mm = double(coords_cell{side});
if size(coords_mm, 2) ~= 3
    error('reco.native.coords_mm{%d} must be an Nx3 numeric array.', side);
end

end

% ========================================================================
function [head_mm, tail_mm] = get_head_tail(reco, side)
% Retrieve head/tail markers for a given side.

markers = reco.native.markers;

if numel(markers) < side
    error('Markers array has only %d entries; requested side index %d.', numel(markers), side);
end

m = markers(side);
if ~isfield(m, 'head') || ~isfield(m, 'tail')
    error('Markers for side %d do not contain required fields "head" and "tail".', side);
end

head_mm = double(m.head(:).');
tail_mm = double(m.tail(:).');

if numel(head_mm) ~= 3 || numel(tail_mm) ~= 3
    error('Head and tail markers must be 1x3 vectors for side %d.', side);
end

end

% ========================================================================
function [scrf_cell, scrf_affine_path] = apply_scrf_affine_if_present(native_cell, sides, opt)
% Apply brainshift affine transform (SCRF) if Lead-DBS provides it.

scrf_affine_path = '';
scrf_cell = native_cell;

if isfield(opt, 'subj') && isfield(opt.subj, 'brainshift') && isfield(opt.subj.brainshift, 'transform') && isfield(opt.subj.brainshift.transform, 'scrf')
    scrf_affine_path = opt.subj.brainshift.transform.scrf;
end

if isempty(scrf_affine_path) || ~isfile(scrf_affine_path)
    return;
end

d = load(scrf_affine_path);
if ~isfield(d, 'mat')
    error('SCRF affine file does not contain variable "mat": %s', scrf_affine_path);
end
A = double(d.mat);
if ~isequal(size(A), [4 4])
    error('SCRF affine matrix must be 4x4: %s', scrf_affine_path);
end

for side = sides
    xyz = native_cell{side};
    if isempty(xyz)
        scrf_cell{side} = zeros(0,3);
        continue;
    end
    scrf_cell{side} = apply_affine_4x4(A, xyz);
end

end

% ========================================================================
function out = apply_affine_4x4(A, xyz)
% Apply 4x4 affine matrix to Nx3 coordinates.

n = size(xyz, 1);
tmp = (A * [xyz, ones(n,1)]')';
out = tmp(:,1:3);

end

% ========================================================================
function [mni_cell, anat_nii_path, transform_folder] = warp_scrf_to_mni(scrf_cell, sides, opt, transform_folder_name)
% Warp SCRF coordinates to MNI using Lead-DBS deformation fields.

if ~isfield(opt, 'subj') || ~isfield(opt.subj, 'coreg') || ~isfield(opt.subj.coreg, 'anat') || ~isfield(opt.subj.coreg.anat, 'preop')
    error('Lead-DBS options missing opt.subj.coreg.anat.preop.');
end
if ~isfield(opt.subj, 'AnchorModality')
    error('Lead-DBS options missing opt.subj.AnchorModality.');
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

if ~isfield(opt.subj, 'subjDir')
    error('Lead-DBS options missing opt.subj.subjDir.');
end

transform_folder = fullfile(opt.subj.subjDir, transform_folder_name);
% if ~isfolder(transform_folder)
%     error('Transform folder not found: %s', transform_folder);
% end

mni_cell = scrf_cell;
for side = sides
    xyz = scrf_cell{side};
    if isempty(xyz)
        mni_cell{side} = zeros(0,3);
        continue;
    end
    mni_cell{side} = warp_mm_coords_to_mni_mm(xyz, nii, transform_folder);
end

end

% ========================================================================
function mni_mm = warp_mm_coords_to_mni_mm(xyz_mm, nii, transform_folder)
% Wrapper around ea_map_coords: mm -> vox -> deformation -> mm.

n = size(xyz_mm, 1);
vox = (nii.mat \ [xyz_mm, ones(n,1)]')';
vox = vox(:,1:3);

% ea_map_coords expects 3xN coordinates.
def = ea_map_coords(vox', nii.fname, transform_folder, '');

mni_mm = def';

end
