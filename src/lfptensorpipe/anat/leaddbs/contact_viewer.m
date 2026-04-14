function contact_viewer(csv_path, atlas_name)
%CONTACT_VIEWER Visualize contact coordinates in MNI space on a Lead-DBS atlas.
%
% contact_viewer(csv_path, atlas_name)
%
% Inputs
% ------
% csv_path
%   Path to the CSV exported by lead.py (df_loc), containing MNI_x/y/z.
% atlas_name
%   Lead-DBS atlas name (used by ea_mnifigure).
%

if nargin < 1 || isempty(csv_path)
    error('csv_path is required.');
end
if nargin < 2 || isempty(atlas_name)
    error('atlas_name is required.');
end

% Open viewer with atlas
try
    ea_mnifigure(atlas_name);
catch
    ea_mnifigure; % fallback if atlas name unsupported
end

main_fig = gcf;
main_ax = gca;

% Add contact coordinates
T = readtable(csv_path);
var_names = T.Properties.VariableNames;
x_col = resolve_var_name(var_names, {'MNI_x', 'mni_x'});
y_col = resolve_var_name(var_names, {'MNI_y', 'mni_y'});
z_col = resolve_var_name(var_names, {'MNI_z', 'mni_z'});
coords = [T.(x_col), T.(y_col), T.(z_col)];
coords(any(isnan(coords),2),:) = [];

radius = 0.127;
n = 100;
color = [0.4, 1, 1];

% draw in main axes
axes(main_ax);
h_spheres = add_spheres(coords, radius, n, 'FaceColor', color);

% Separate small QC window
qc_fig = figure('Name', 'Sphere', ...
    'NumberTitle', 'off', ...
    'MenuBar', 'none', ...
    'ToolBar', 'none', ...
    'Units', 'pixels', ...
    'Position', [100 100 260 140]);

panel = uipanel('Parent', qc_fig, ...
    'Title', 'Sphere', ...
    'Units', 'normalized', ...
    'Position', [0.05 0.08 0.9 0.88]);

uicontrol('Parent', panel, 'Style', 'text', ...
    'String', 'Radius', ...
    'Units', 'normalized', ...
    'HorizontalAlignment', 'left', ...
    'Position', [0.06 0.62 0.4 0.2]);

radius_slider = uicontrol('Parent', panel, 'Style', 'slider', ...
    'Min', 0.01, 'Max', 1.0, 'Value', radius, ...
    'Units', 'normalized', ...
    'Position', [0.06 0.48 0.88 0.15], ...
    'Callback', @onRadiusChange);

radius_edit = uicontrol('Parent', panel, 'Style', 'edit', ...
    'String', sprintf('%.3f', radius), ...
    'Units', 'normalized', ...
    'Position', [0.60 0.66 0.34 0.2], ...
    'Callback', @onRadiusEdit);

uicontrol('Parent', panel, 'Style', 'pushbutton', ...
    'String', 'Pick Color', ...
    'Units', 'normalized', ...
    'Position', [0.06 0.15 0.88 0.22], ...
    'Callback', @onPickColor);

    function update_spheres()
        if isgraphics(h_spheres)
            delete(h_spheres);
        end
        if isgraphics(main_fig)
            figure(main_fig);
            axes(main_ax);
            h_spheres = add_spheres(coords, radius, n, 'FaceColor', color);
        end
    end

    function onRadiusChange(src, ~)
        radius = get(src, 'Value');
        set(radius_edit, 'String', sprintf('%.3f', radius));
        update_spheres();
    end

    function onRadiusEdit(src, ~)
        val = str2double(get(src, 'String'));
        if isnan(val)
            set(src, 'String', sprintf('%.3f', radius));
            return;
        end
        val = max(get(radius_slider, 'Min'), min(get(radius_slider, 'Max'), val));
        radius = val;
        set(radius_slider, 'Value', radius);
        set(src, 'String', sprintf('%.3f', radius));
        update_spheres();
    end

    function onPickColor(~, ~)
        c = uisetcolor(color, 'Pick sphere color');
        if isequal(size(c), [1 3])
            color = c;
            update_spheres();
        end
    end

end

function resolved = resolve_var_name(var_names, candidates)
% Resolve coordinate column names with case/legacy compatibility.
resolved = '';
for i = 1:numel(candidates)
    idx = strcmp(var_names, candidates{i});
    if any(idx)
        resolved = var_names{find(idx, 1, 'first')};
        return;
    end
end
for i = 1:numel(candidates)
    idx = strcmpi(var_names, candidates{i});
    if any(idx)
        resolved = var_names{find(idx, 1, 'first')};
        return;
    end
end
error('Missing coordinate column. Expected one of: %s', strjoin(candidates, ', '));
end

function h = add_spheres(centers, radius, n, varargin)
% centers: [N x 3] (mm, same space as atlas mesh)
% radius : scalar (mm)
% n      : sphere mesh resolution
% varargin: passed to patch

if isempty(centers)
    h = gobjects(0, 1);
    return;
end
if nargin < 3 || isempty(n), n = 12; end
if nargin < 2, error('Need centers and radius'); end

% base unit sphere -> triangles
[X, Y, Z] = sphere(n);
[F0, V0] = surf2patch(X * radius, Y * radius, Z * radius, 'triangles');
nv = size(V0, 1);
nf = size(F0, 1);

k = size(centers, 1);
V = zeros(nv * k, 3);
F = zeros(nf * k, 3);

for i = 1:k
    vi = (i - 1) * nv + (1:nv);
    fi = (i - 1) * nf + (1:nf);
    V(vi, :) = V0 + centers(i, :);
    F(fi, :) = F0 + (i - 1) * nv;
end

hold on
h = patch('Faces', F, 'Vertices', V, ...
          'FaceColor', [1 1 1], 'EdgeColor', 'none', ...
          'FaceAlpha', 1, varargin{:});
lighting gouraud; material dull; camlight('headlight');
end
