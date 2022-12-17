function vox = tightestBoundary(vox_input,shrink,mode,visualizeLogic)
%Input:     input voxel shape [MxNxL] 
%           intensity of voxel always "1"
%           shrink = 0.0(convex hull) to 1.0(compact boundary)
%           mode = 1 find boundary of all pixel in slice
%           mode = 2 find boundary of each connected object in slice
%           visualizeLogic = 1 visualize boundary by subplot
%           visualizeLogic = 2 visualize boundary by plot each slice
%           visualizeLogic = 0 not visualize

%Output:    voxel that was perform "tightestBoundary"
%           output have same size of input
%
% For Femoral bone segmentation
%
% Implemented by Danupong Buttongkum (FEW Assassin) 01-Aug-2020


%% Main Program
pcdVox = vox2pcd(vox_input);
vox = false(size(vox_input));
[sy sx sz] = size(vox);
minK = min(pcdVox(:,3));
maxK = max(pcdVox(:,3));
subplotArray = ceil((sqrt(maxK-minK)));


if mode == 1
    % Mode = 1 : find boundary of all pixel in slice
    parfor k = minK:maxK
        idz = find(pcdVox(:,3)==k);
        % 
        if isempty(idz)==0
            pcd_slice_k = pcdVox(idz,:);    % [Nx4] at slice 'k'
            pcd_x = pcd_slice_k(:,1); 
            pcd_y = pcd_slice_k(:,2);
            pcd_z = pcd_slice_k(:,3);
            bw = boundary(pcd_x,pcd_y,shrink);   
            bw_point_x = pcd_x(bw);
            bw_point_y = pcd_y(bw);
            bw_point_z = pcd_z(bw);
            
            if visualizeLogic==1   % visual by subplot
                subplot(subplotArray,subplotArray,k-minK+1); plot(pcd_x,pcd_y,'.','Color' ,'blue')
                xlim([min(pcdVox(:,1)) max(pcdVox(:,1))])
                ylim([min(pcdVox(:,2)) max(pcdVox(:,2))])
                hold on
                subplot(subplotArray,subplotArray,k-minK+1);plot(bw_point_x,bw_point_y)
                titleStr = strcat('Slice NO. : ',string(k));
                title(titleStr)
            elseif visualizeLogic==2  % visual by plot each layer
                plot(pcd_x,pcd_y,'.','Color' ,'blue')
                xlim([min(pcdVox(:,1)) max(pcdVox(:,1))])
                ylim([min(pcdVox(:,2)) max(pcdVox(:,2))])
                hold on
                plot(bw_point_x,bw_point_y)
                titleStr = strcat('Slice NO. : ',string(k));
                pause(0.75)
            end
            if visualizeLogic==2
                close
            end
            
            % creating mask and vox
            mask = poly2mask(bw_point_x, bw_point_y, sy, sx);
            vox(:,:,k) = mask;
        end
    end

elseif mode == 2
    % Mode = 2 : find boundary of each connected object in slice
    subplotArray = ceil((sqrt(maxK-minK)));
    parfor k = minK:maxK
        idz = find(pcdVox(:,3)==k);
        vox_k = imclose(vox_input(:,:,k),strel('disk',2));
        if isempty(idz)==0
            CC = bwconncomp(vox_input(:,:,k));   % slice k 
            %showText = strcat('Slice NO.',string(k),'   NumObjects = ',string(CC.NumObjects));
            %fprintf(showText); fprintf('\n')
            mask = false(size(vox_input,1),size(vox_input,2));
            for c = 1:CC.NumObjects     % find boundary of each object in slice k
                %showText = strcat(' - Process Obj: ',string(c),'  \n');
                %fprintf(showText);
                obj = false(size(vox_input(:,:,k)));
                obj(CC.PixelIdxList{1, c}) = 1;
                
                % Pixel point
                pcdObj = vox2pcd(obj);    % [Nx4] at slice 'k'
                pcdObj_x = pcdObj(:,1);
                pcdObj_y = pcdObj(:,2);
                pcdObj_z = pcdObj(:,3);
                
                % Boundary point
                bwObj = boundary(pcdObj_x,pcdObj_y,shrink);   
                bwObj_x = pcdObj_x(bwObj);
                bwObj_y = pcdObj_y(bwObj);
                bwObj_z = pcdObj_z(bwObj);
                
                if visualizeLogic==1
                    hold on
                    subplot(subplotArray,subplotArray,k-minK+1); plot(pcdObj_x,pcdObj_y,'.','Color' ,'blue');
                    xlim([min(pcdVox(:,1)) max(pcdVox(:,1))]);
                    ylim([min(pcdVox(:,2)) max(pcdVox(:,2))]);
                    hold on
                    subplot(subplotArray,subplotArray,k-minK+1); plot(bwObj_x,bwObj_y,'red');
                    titleStr = strcat('NO.',string(k));
                    title(titleStr)
                elseif visualizeLogic==2
                    hold on
                    plot(pcdObj_x,pcdObj_y,'.','Color' ,'blue');
                    xlim([min(pcdVox(:,1)) max(pcdVox(:,1))]);
                    ylim([min(pcdVox(:,2)) max(pcdVox(:,2))]);
                    hold on
                    plot(bwObj_x,bwObj_y,'red');
                    titleStr = strcat('Slice NO. : ',string(k));
                    title(titleStr)
                    pause(0.5);
                end
                
                % creating mask and vox of each slice k
                mask_k = poly2mask(bwObj_x, bwObj_y, sy, sx);
                mask = mask|mask_k;    % union opperation
            end
            if visualizeLogic==2
                close
            end
            vox(:,:,k) = mask;
        end
    end
else
    fprintf('   *** Mode is not correct ... \n')
end












%% Convert input voxel to point cloud format

%{
pcdBoundary = boundary(pcdVox(:,1),pcdVox(:,2),pcdVox(:,3),1);
% Find boundaryPoint on boundary
uniqueBoundary = sort(unique(pcdBoundary(:)));
boundaryPoint = pcdVox(uniqueBoundary,:);
a = boundaryPoint(:,1); b = boundaryPoint(:,2); 
c = boundaryPoint(:,3);d = boundaryPoint(:,4);

figure; plot3(a,b,c,'o')
grid on;
visualBoundaryLogic = input('   Do you want to visual 3D boundary? [y/n] : ','s');
if visualBoundaryLogic=='y'
    hold on; 
    trisurf(pcdBoundary,pcdVox(:,1),pcdVox(:,2),pcdVox(:,3),'Facecolor','red','FaceAlpha',0.1);
end

%idx = sub2ind(size(vox),(1:numel(boundaryPoint)).',boundaryPoint);
%}

%%

% 'blendedPolymask' function
%{
X = linspace(1, size(femurL,1),size(femurL,1));
Y = linspace(1, size(femurL,2),size(femurL,2));
Z = linspace(1,size(femurL,3),size(femurL,3));
C = {};
for i = 1:max(pcdVox(:,3))
    C = cat(1,C,pcdVox(find(pcdVox(:,3)==i),1:3));  % Concatenate array
end
BW = blendedPolymask(C,X,Y,Z);
%}

% Demo 'blendedPolymask' function
%{
[circXY(:,1),circXY(:,2)] = pol2cart(linspace(0,2*pi,50)', 1);
sqXY = [-1 -1;1 -1;1 1;-1 1; -1 -1];
C = {[sqXY*5 ones(5,1)]           % Start with a small square
    [circXY*40 ones(50,1)*30]     % Blend to a large circle
   [sqXY*20 ones(5,1)*65]        % Blend to a large square
   [circXY*10 ones(50,1)*99]};   % Blend to a small circle
X = linspace(-40, 40, 200);
Y = linspace(-40, 40, 200);
Z = linspace(0, 100, 400);
BW = blendedPolymask(C,X,Y,Z);
figure, patch(isosurface(X,Y,Z,BW,0.5),'FaceColor','g','EdgeColor','none','FaceAlpha',0.5)
view(3), camlight, hold on, axis image
cellfun(@(x)patch(x(:,1),x(:,2),x(:,3),'b'),C)
%}


end