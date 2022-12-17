%% Data Augmentation %%
% 
% For augmentation and synthesizing auxiliary class
% 
% Last Modified Date: 01-05-2022
% On device: Tyan-GPU
% 
%% Initial
startLogic = input('   Starting the process? [y/n]: ','s');
if startLogic=='y'
    clear all
    close all
    volumeViewer close
    clc
    opengl hardware
    first = 1;
end
screensize = get(0,'Screensize');

%% Download all
path = uigetdir('Select RAW Data file of specific sample ');
dataset = dir(path);

%% Start for loop

for i = 3:size(dataset,1)-1  % Start at i = 3
    %% Load RAW data
    if first==1
        pauseEachSample = input('   Waiting for [Enter] to go to the next sample? [y/n] : ','s');
    end
    timeStart = tic;
    file = dataset(i).name;
    address = strcat(path,'\',file);
    load(address)
    try
        m = info.RescaleSlope;
    catch
        fprintf('   Warning! RescaleSlope is not found\n')
        m = 1.0;
    end
    try
        b = info.RescaleIntercept;
    catch
        fprintf('   Warning! RescaleIntercept is not found\n')
        b = -1024.0;
    end
    fprintf('\n\n\n#############################################################\n')
    fprintf('   ########## DICOM Attribute ##########  \n');
    fprintf('   File Name = %s \n', address)
    fprintf('   Accession Number : %s \n', info.AccessionNumber);
    fprintf('   Rescale Slope = %f \n',m);
    fprintf('   Rescale Intercept = %f \n',b);
    fprintf('   sfactors = [%f %f %f] \n', sfactors);
    fprintf('   Fracture Type = [%u %u] \n', fractureType);
    
    % Setting 'volshow' configuration
    config = struct;
    config.ScaleFactors = sfactors; 
    config.BackgroundColor = [114 200 100]./255;
    config.Isovalue = 0.4;
    config.Renderer = 'VolumeRendering';
    config.CameraPosition = [0 -1 0];
    
    config2 = struct;
    config2.ScaleFactors = [1 1 1]; 
    config2.BackgroundColor = [114 200 100]./255;
    config2.Isovalue = 0.4;
    config2.Renderer = 'VolumeRendering';
    config2.CameraPosition = [0 -1 0];
    
    config3 = struct;
    config3.ScaleFactors = [1 1 1]; 
    config3.BackgroundColor = [114 200 100]./255;
    config3.Isovalue = 0.4;
    config3.Renderer = 'VolumeRendering';
    config3.CameraPosition = [1 0 0];
    
    % Check for Conformity of RAW Data
    fprintf('   Volume Size = [%u %u %u] \n', size(v));
    if floor(size(v,3).*sfactors(3)/sfactors(1)) < 256
        fprintf('Not enough Z-dim >>> Go to the next sample')
        continue;           % skip this loop i
    end
    
    if first==1
        showRawLogic = input('   Do you want to visual Completed mask (Raw scale)? [y/n] : ','s');
    end
    if showRawLogic=='y'    
        volumeViewer(v, mask, 'ScaleFactors', sfactors);
    end
    
    
    %% Retieve anatomy and Synthesizing auxiliary class
    
    v = int16(v);
    pelvicMask = uint8(false(size(v)));         femurRMask = uint8(false(size(v)));
    femurLMask = uint8(false(size(v)));         softMask = uint8(false(size(v)));
    fractureMask = uint8(false(size(v)));       fractureMask0 = uint8(false(size(v)));
    compressMask = uint8(false(size(v)));
    pelvicMask(ismember(mask,[20])) = 1;        pelvicMask = uint8(bwareaopen(pelvicMask, 20));
    compressMask(ismember(mask,[50])) = 1;     compressMask = uint8(bwareaopen(compressMask, 10));
    
    % Skip if softMask is conformity
    % Segment only soft-tissue
    if b == 0
        fprintf('   Warning! : Rescale Intercept = 0 !!!\n')
        v = v + 1024;
        softMask(ismember(mask,[30])) = 1;
        softMask = uint8(bwareaopen(softMask, 20));
        figure; sliceViewer(softMask);
        %soft_logic = input('   Select soft masking method [1] use softMask  [2] tightestBoundary = ');
        soft_logic = 1;
        if soft_logic==1
            tic
            parfor s = 1:size(softMask,3)
                softMask(:,:,s) = imfill(softMask(:,:,s),'holes');
            end
            softMask = uint8(softMask);
            figure; sliceViewer(softMask);
            toc
            fprintf('   [Enter] to next \n');
            %pause();
        else
            tic
            softMask = tightestBoundary(softMask, 0.85, 2, 0);    % มีปัญหา  Array construction from ByteBuffer Threw an exception  น่าจะ out of memory
            softMask = uint8(softMask);
            figure; sliceViewer(softMask);
            toc
            fprintf('   [Enter] to next \n');
            pause();
        end
    else
        f_threshold = 276 - b;
        b_threshold = -324 - b;
        L_soft = (v>b_threshold) & (v<f_threshold);
        CC_soft = bwconncomp(L_soft);
        numPixels_soft = cellfun(@numel,CC_soft.PixelIdxList);
        [biggest_soft,idx_soft] = max(numPixels_soft);        % find maximum connected voxel 
        softMask(CC_soft.PixelIdxList{1,idx_soft}) = 1;
        parfor s = 1:size(softMask,3)
            softMask(:,:,s) = imfill(softMask(:,:,s),'holes');
        end
    end
    
    c2 = [1,3,5,7,9,11,13];    % femurR_member = [1,3,5,7,9,11,13,15,17]
    c3 = [2,4,6,8,10,12,14];   % femurL_member = [2,4,6,8,10,12,14,16,18]
    dilateR = 0;    dilateL = 0;
    if fractureType(1)==1
        dilateR = 1;
    elseif fractureType(1)==2
        dilateR = 2;
    end
    
    if fractureType(2)==1
        dilateL = 1;
    elseif fractureType(2)==2
        dilateL = 2;
    end
    
    % Synthesize auxiliary class
    for ri = 1:size(c2,2)    % Right
        femurRMask = femurRMask + ri.*uint8(bwareaopen(mask==c2(1,ri), 20));
        fractureMask = fractureMask + uint8(imdilate(femurRMask==ri, strel('sphere',dilateR)));
    end
    for li = 1:size(c3,2)    % Left
        femurLMask = femurLMask + li.*uint8(bwareaopen(mask==c3(1,li), 20));
        fractureMask = fractureMask + uint8(imdilate(femurLMask==li, strel('sphere',dilateL)));
    end
    fractureMask = uint8(fractureMask>1);
    %figure; sliceViewer(fractureMask); title('fractureMask');
    
    fractureMask0(ismember(mask, [40])) = 1;
    fractureMask0 = uint8(bwareaopen(fractureMask0, 10));
    %figure; sliceViewer(fractureMask0); title('fractureMask0');
    
    fractureMask = uint8( (fractureMask + fractureMask0)>0 );
    %figure; sliceViewer(fractureMask); title('fractureMask Final');
    
    fractureMask0 = fractureMask;
    % Create fractureMask
    fractureMask = fractureMask - uint8(femurRMask>1) - uint8(femurLMask>1);      % remove intersection of fracture and smaller femur's fragment 
    femurRMask = femurRMask.*( uint8(femurRMask>0) - fractureMask );              % remove femurR and fracture intersection
    femurLMask = femurLMask.*( uint8(femurLMask>0) - fractureMask );              % remove femurR and fracture intersection
    softMask = softMask - pelvicMask-femurRMask - femurLMask -fractureMask - compressMask;
    
    if first==1
        showRawLogic = input('   Do you want to visual Completed mask after simulate fracture (Raw scale)? [y/n] : ','s');
    end
    if showRawLogic=='y'
        close all
        volumeViewer close
        volumeViewer(v, (femurRMask.*2-1).*uint8(femurRMask>0) + femurLMask.*2 + pelvicMask.*20 + softMask.*30 + uint8(fractureMask).*40 + uint8(compressMask).*50, 'ScaleFactors',sfactors);
    end
    
    clear b_threshold f_threshold L_soft CC_soft numPixels_soft biggest_soft idx_soft
    
    %% Normalization Volume
    
    fprintf('   Refine Volume  \n');
    fprintf('   ------------------------------------------------------ \n');
    fprintf('   Resolution = [%f %f %f] mm\n', sfactors);
    fprintf('   Original volume = [%u %u %u] voxels\n', size(v));
    fprintf('   True-scale of original volume = [%u %u %u] voxels\n', round(size(v).*sfactors./sfactors(1)));
    fprintf('      VSize(mm) = [%f %f %f] mm   (check!) \n', size(v).*sfactors);
    if first==1
        resize_mode = input('   Resize mode : [1] True-scale as sfactors  [2] Normalization = ');
        if resize_mode == 2
            resolution = input('   Input Resolution = ');
        end
    end
    
    if resize_mode == 1
        fprintf(' Resize mode : [1] True-scale as sfactors >>> ');
        if sfactors(1) < resolution   % for only very fine resolution sample 
            fprintf('Condition1 : smaller than threshold \n');
            upscale = (sfactors(1)./resolution);
            sfactors2 = [resolution resolution resolution];
        else
            fprintf('Condition2 : larger than threshold \n');
            upscale = 1.0;
            sfactors2 = [sfactors(1) sfactors(1) sfactors(1)];
        end
    elseif resize_mode == 2
        fprintf('   Resize mode : [2] Normalization \n');
        upscale = (sfactors(1)/resolution);
        sfactors2 = [resolution resolution resolution];
    else
        fprintf('   resize_mode param. error!!! \n');
        wtf
    end
    Vsize2 = round(size(v).*sfactors./sfactors(1).*upscale);
    fprintf('   ------------------------------------------------------ \n');
    fprintf('   Adjusted resolution = [%f %f %f] mm\n', sfactors2);
    fprintf('   Adjusted true-scale volume = [%u %u %u] voxels \n', Vsize2);
    fprintf('      VSize(mm) = [%f %f %f] mm   (check!) \n', Vsize2.*sfactors2);
    fprintf('   ------------------------------------------------------ \n');
    config2.ScaleFactors = sfactors2;    % AP-view
    config3.ScaleFactors = sfactors2;    % Lateral-view
    saveCentroid2 = saveCentroid;
    saveCentroid2(1) = saveCentroid(1)*upscale;
    saveCentroid2(2) = saveCentroid(2)*upscale;
    saveCentroid2(4) = saveCentroid(4)*upscale;
    saveCentroid2(5) = saveCentroid(5)*upscale;
    saveCentroid2(3) = saveCentroid(3)*sfactors(3)/(sfactors(1))*upscale;
    saveCentroid2(6) = saveCentroid(6)*sfactors(3)/(sfactors(1))*upscale;
    tic
    v2 = imresize3(v,Vsize2,'cubic');
    pelvicMask2 = imresize3(pelvicMask,Vsize2,'nearest');
    femurLMask2 = imresize3(femurLMask,Vsize2,'nearest');
    femurRMask2 = imresize3(femurRMask,Vsize2,'nearest');
    softMask2 = imresize3(softMask,Vsize2,'nearest');
    fractureMask2 = imresize3(fractureMask,Vsize2,'nearest');
    compressMask2 = imresize3(compressMask, Vsize2, 'nearest');
    toc
    if size(v2,3) < 256
        fprintf('Not enough Z-dim >>> Go to the next sample')
        continue;           % skip this loop i
    end
    if first==1
        showRawLogic = input('   Do you want to visual Normalized mask? [y/n] : ','s');
    end
    if showRawLogic=='y'
        close all
        volumeViewer close
        volumeViewer(v2, (femurRMask2.*2-1).*uint8(femurRMask2>0) + femurLMask2.*2 + pelvicMask2.*20 + softMask2.*30 + uint8(fractureMask2).*40 + uint8(compressMask2).*50, 'ScaleFactors',sfactors2);
    end
    
    clear pelvicMask femurLMask femurRMask softMask fractureMask comminuteMask softMask v    % remove old variable to save RAM space
    
    %% Remove Intersection Mask
    
    fprintf('   Remove intersection bone \n' );
    tic
    pelvicMask2 = pelvicMask2 - uint8(femurRMask2>0) - uint8(femurLMask2>0) - uint8(fractureMask2) - uint8(compressMask2);
    fractureMask2 = fractureMask2 - uint8(femurRMask2>0) - uint8(femurLMask2>0);
    compressMask2 = compressMask2 - uint8(femurRMask2>0) - uint8(femurLMask2>0) - fractureMask2;
    softMask2 = softMask2 - uint8(femurRMask2>0) - uint8(femurLMask2>0) - fractureMask2 - compressMask2;
    toc
    
    if first==1
        showTrueScaleLogic = input('   Do you want to visual Completed normalized mask? [y/n] : ','s');
    end
    if showTrueScaleLogic=='y'
        close all
        volumeViewer close
        volumeViewer(v2,(femurRMask2.*2-1).*uint8(femurRMask2>0) + femurLMask2.*2 + pelvicMask2.*20 + softMask2.*30 + uint8(fractureMask2).*40 + uint8(compressMask2).*50, 'ScaleFactors',sfactors2);
    end
    
    
    %% Start Augmentation Loop
    angleList = [-45];
    for p = 1:size(angleList,2)
        %% Rotate and Flip Volume
        % for FemurSide = 1:2   % loop for left and right-side >>> future
        % work
        fprintf('   ## Rotate and Flip volume \n');
        angle = angleList(p);
        tic
        angle_rad = angle*pi/180.0;
        
        v30 = v2;            % Left side femur
        pelvicMask30 = pelvicMask2;
        femurRMask30 = uint8(femurRMask2);
        femurLMask30 = uint8(femurLMask2);
        softMask30 = softMask2;
        fractureMask30 = fractureMask2;             % For Fracture sample
        v40 = flip(v2,2);    % Right side femur
        pelvicMask40 = flip(pelvicMask2,2);
        femurRMask40 = flip(uint8(femurRMask2),2);
        femurLMask40 = flip(uint8(femurLMask2),2);
        softMask40 = flip(softMask2,2);
        fractureMask40 = flip(fractureMask2,2);     % For Fracture sample
        
        bbox = 'loose';
        if sqrt(size(v2,1)^2 + size(v2,2)^2) < 600
            bbox = 'loose';
        end
        if sqrt(size(v2,1)^2 + size(v2,2)^2) > 1000
            bbox = 'crop';
        end
        %bbox = 'loose';
        fprintf('        bbox = %s \n', bbox);
        parfor k = 1:size(v2,3)       % future work >>> use "imrotate3_fast" function instead
            % Rotate Left-side
            dum1(:,:,k) = imrotate(v30(:,:,k),angle,'bicubic',bbox);
            dum2(:,:,k) = imrotate(pelvicMask30(:,:,k),angle,'nearest',bbox);
            dum3(:,:,k) = imrotate(femurRMask30(:,:,k),angle,'nearest',bbox);
            dum4(:,:,k) = imrotate(femurLMask30(:,:,k),angle,'nearest',bbox);
            dum5(:,:,k) = imrotate(softMask30(:,:,k),angle,'nearest',bbox);
            dum6(:,:,k) = imrotate(fractureMask30(:,:,k),angle,'nearest',bbox);    % For Fracture sample
            % Rotate Right-side
            dum7(:,:,k)  = imrotate(v40(:,:,k),angle,'bicubic',bbox);
            dum8(:,:,k)  = imrotate(pelvicMask40(:,:,k),angle,'nearest',bbox);
            dum9(:,:,k)  = imrotate(femurRMask40(:,:,k),angle,'nearest',bbox);
            dum10(:,:,k) = imrotate(femurLMask40(:,:,k),angle,'nearest',bbox);
            dum11(:,:,k) = imrotate(softMask40(:,:,k),angle,'nearest',bbox);
            dum12(:,:,k) = imrotate(fractureMask40(:,:,k),angle,'nearest',bbox);   % For Fracture sample
        end
        v30 = dum1; pelvicMask30 = dum2; femurRMask30 = dum3; femurLMask30 = dum4; softMask30 = dum5; fractureMask30 = dum6;
        v40 = dum7; pelvicMask40 = dum8; femurRMask40 = dum9; femurLMask40 = dum10; softMask40 = dum11; fractureMask40 = dum12;
        clear dum1 dum2 dum3 dum4 dum5 dum6 dum7 dum8 dum9 dum10 dum11 dum12
        
        % Original Centroids which are used again for fracture simulation
        center = [double(size(v2,2))/2.0 double(size(v2,1))/2.0];           % [X-axis Y-axis]
        center2 = [double(size(v30,2))/2.0 double(size(v30,1))/2.0];        % [X-axis Y-axis]
        centroid1 = [saveCentroid2(4) saveCentroid2(5) saveCentroid2(6)];
        centroid2 = [(size(v2,2)-saveCentroid2(1)) saveCentroid2(2) saveCentroid2(3)];  % flip
        R1 = sqrt((centroid1(1)-center(1)).^2+(centroid1(2)-center(2)).^2);
        theta1 = atan((centroid1(2)-center(2))/(centroid1(1)-center(1)));
        R2 = sqrt((centroid2(1)-center(1)).^2+(centroid2(2)-center(2)).^2);
        theta2 = atan((centroid2(2)-center(2))/(centroid2(1)-center(1)));
        
        % Calculate centroid after rotation
        centroid1(1) = R1*cos(-angle_rad+theta1)+center2(1);   % Left femur X-axis
        centroid1(2) = R1*sin(-angle_rad+theta1)+center2(2);   % Left femur Y-axis
        centroid2(1) = R2*cos(-angle_rad+theta2)+center2(1);   % Right femur X-axis
        centroid2(2) = R2*sin(-angle_rad+theta2)+center2(2);   % Right femur Y-axis

        R1 = sqrt((centroid1(1)-center(1)).^2+(centroid1(2)-center(2)).^2);
        R2 = sqrt((centroid2(1)-center(1)).^2+(centroid2(2)-center(2)).^2);
        centroid1 = round(centroid1);   centroid2 = round(centroid2);
        toc
        
        % Get even array
        Vsize3 = size(v30);
        v30 = v30(1:end-mod(Vsize3(1),2) , 1:end-mod(Vsize3(2),2) , :);
        pelvicMask30 = pelvicMask30(1:end-mod(Vsize3(1),2) , 1:end-mod(Vsize3(2),2) , :);
        femurRMask30 = femurRMask30(1:end-mod(Vsize3(1),2) , 1:end-mod(Vsize3(2),2) , :);
        femurLMask30 = femurLMask30(1:end-mod(Vsize3(1),2) , 1:end-mod(Vsize3(2),2) , :);
        softMask30 = softMask30(1:end-mod(Vsize3(1),2) , 1:end-mod(Vsize3(2),2) , :);
        fractureMask30 = fractureMask30(1:end-mod(Vsize3(1),2) , 1:end-mod(Vsize3(2),2) , :);
        Vsize4 = size(v40);
        v40 = v40(1:end-mod(Vsize4(1),2) , 1:end-mod(Vsize4(2),2) , :);
        pelvicMask40 = pelvicMask40(1:end-mod(Vsize4(1),2) , 1:end-mod(Vsize4(2),2) , :);
        femurRMask40 = femurRMask40(1:end-mod(Vsize4(1),2) , 1:end-mod(Vsize4(2),2) , :);
        femurLMask40 = femurLMask40(1:end-mod(Vsize4(1),2) , 1:end-mod(Vsize4(2),2) , :);
        softMask40 = softMask40(1:end-mod(Vsize4(1),2) , 1:end-mod(Vsize4(2),2) , :);
        fractureMask40 = fractureMask40(1:end-mod(Vsize4(1),2) , 1:end-mod(Vsize4(2),2) , :);
        whos v30 pelvicMask30 femurRMask30 femurLMask30 softMask30 fractureMask30
        whos v40 pelvicMask40 femurRMask40 femurLMask40 softMask40 fractureMask40
        
        
        if first==1
            showCropBoundaryLogic = input('   Do you want to visual crop boundary? [y/n] : ','s');
        end
        if showCropBoundaryLogic=='y'
            bone3 = uint8(pelvicMask30).*20+uint8(femurLMask30)+uint8(softMask30).*30+uint8(fractureMask30).*40;
            bone4 = uint8(pelvicMask40).*20+uint8(femurRMask40)+uint8(softMask40).*30+uint8(fractureMask40).*40;
            threshold1 = 10;  % shift in X-axis   % future work >>> random number of the threshold
            threshold2 = 10;  % shift in Y-axis   % future work >>> random number of the threshold
            figure; sliceViewer(bone3);
            crop1 = [centroid1(1)-127+threshold1 centroid1(2)-127+threshold2 256 256];
            crop2 = [centroid2(1)-127+threshold1 centroid2(2)-127+threshold2 256 256];
            drawrectangle('Position',crop1,'InteractionsAllowed','none');
            figure; sliceViewer(bone4);
            drawrectangle('Position',crop2,'InteractionsAllowed','none');
            pause(15);
        end
        
        clear bone3 bone4 crop1 crop2 threshold1 threshold2
        
        %% Simulated Fracture of Bone
        % Input = [v3,v4,pelvicMask3,pelvicMask4,femurRMask3,femurLMask4
        %          softMask3,softMask4,centroid1,centroid2]
        
        for k = 0:7   % k: 0=Non-fracture  1-7=Fracture

            augmentNO = sprintf('%02d', 20*(p-1) + k );   % for fracture
            fprintf('\n   ### Augmentation NO. = %s   | Angle rotation = %f \n',augmentNO, angle);
            
            %% Randomly simulate fracture of Bone
            threshold = 10;
            cropOffset1 = -127+threshold;      % set lower limit [ left front ]
            cropOffset2 = +128+threshold;      % set upper limit [ right back ]
            cropOffsetZ = round((saveCentroid2(3) + saveCentroid2(6))/2);
            if cropOffsetZ-169 > 0  && cropOffsetZ+86 <= size(v2,3)   % crop middle of the volume
                fprintf('Condition 1\n')
                cropZ1 = cropOffsetZ-169;
                cropZ2 = cropOffsetZ+86;
            elseif cropOffsetZ-169 <= 0                               % crop lower of the volume
                fprintf('Condition 2\n')
                cropZ1 = 1;
                cropZ2 = 256;
            elseif cropOffsetZ+86 > size(v2,3)                        % crop upper of the volume
                fprintf('Condition 3\n')
                cropZ1 = size(v2,3)-255;
                cropZ2 = size(v2,3);
            end
            
            if k == 0   % Don't simulate fracture
                fprintf('   Without Simulation of Fracture \n')
                tic
                % Crop intensity volume
                % Crop Left-side
                pelvicMask3_   = pelvicMask30(   centroid1(2)+cropOffset1:centroid1(2)+cropOffset2 , centroid1(1)+cropOffset1:centroid1(1)+cropOffset2 , cropZ1:cropZ2 );
                femurLMask3_   = femurLMask30(   centroid1(2)+cropOffset1:centroid1(2)+cropOffset2 , centroid1(1)+cropOffset1:centroid1(1)+cropOffset2 , cropZ1:cropZ2 );
                fractureMask3_ = fractureMask30( centroid1(2)+cropOffset1:centroid1(2)+cropOffset2 , centroid1(1)+cropOffset1:centroid1(1)+cropOffset2 , cropZ1:cropZ2 );
                % Crop Right-side
                pelvicMask4_   = pelvicMask40(   centroid2(2)+cropOffset1:centroid2(2)+cropOffset2 , centroid2(1)+cropOffset1:centroid2(1)+cropOffset2 , cropZ1:cropZ2 );
                femurRMask4_   = femurRMask40(   centroid2(2)+cropOffset1:centroid2(2)+cropOffset2 , centroid2(1)+cropOffset1:centroid2(1)+cropOffset2 , cropZ1:cropZ2 );
                fractureMask4_ = fractureMask40( centroid2(2)+cropOffset1:centroid2(2)+cropOffset2 , centroid2(1)+cropOffset1:centroid2(1)+cropOffset2 , cropZ1:cropZ2 );
                toc
                
                fprintf('   *** Extract intensity volume (Output) *** \n');
                tic
                % Full size volume >>> These variables are used for simulation of DRR-image
                frag3 = max(unique(uint8(femurLMask30)));
                frag4 = max(unique(uint8(femurRMask40)));
                fprintf('   NO. fragment3 = %u \n   NO. fragment4 = %u \n', frag3, frag4);
                boneInten3 = int16(pelvicMask30|femurLMask30|femurRMask30).*v30;
                boneInten4 = int16(pelvicMask40|femurRMask40|femurLMask40).*v40;
                softInten3 = int16(softMask30|fractureMask30).*v30;
                softInten4 = int16(softMask40|fractureMask40).*v40;
                boneMask3  = uint8(pelvicMask30).*20 + uint8(femurLMask30);
                boneMask4  = uint8(pelvicMask40).*20 + uint8(femurRMask40);
                airMask3   = logical((pelvicMask30|femurRMask30|femurLMask30|softMask30|fractureMask30));
                airMask4   = logical((pelvicMask40|femurRMask40|femurLMask40|softMask40|fractureMask40));
                
                % Cropped volume >>> Output Intensity & Mask
                % boneMask = femur1+femur2+femur3+....+pelvic+fracture {uint8}
                boneMask3_ = uint8(femurLMask3_) + uint8(pelvicMask3_).*20 + uint8(fractureMask3_).*40;    % Output: boneMask3 + fractureMask3
                boneMask4_ = uint8(femurRMask4_) + uint8(pelvicMask4_).*20 + uint8(fractureMask4_).*40;    % Output: boneMask4 + fractureMask4
                %affine = zeros(2,6);                                                                      % Output: Affine
                toc
                
            elseif k > 0    % Simulate fracture
                % Cropped volume >>> Output Intensity & Mask
                fprintf('   ### Simulate Fracture Surface ### \n')
                if fractureType(2) == 0      % For only intact left side !!!
                    % Simulate fracture
                    boneMask3 = uint8(pelvicMask30).*1+uint8(femurLMask30).*2;        % for simFracture2  
                    [boneInten3,boneMask3,softInten3,softMask3,fractureMask3,frag3] = simFracture2(v30,boneMask3,softMask30,centroid1,cropZ1,k,angle);
                    %fprintf('   NO. fragment3 = %u \n', frag3);
                    airMask3 = boneMask3|softMask3|fractureMask3;
                    % Cropped volume >>> Output Intensity & Mask
                    boneInten3_    = boneInten3(    centroid1(2)+cropOffset1:centroid1(2)+cropOffset2 , centroid1(1)+cropOffset1:centroid1(1)+cropOffset2 , cropZ1:cropZ2 );       % Output: Intensity3
                    boneMask3_     = boneMask3(     centroid1(2)+cropOffset1:centroid1(2)+cropOffset2 , centroid1(1)+cropOffset1:centroid1(1)+cropOffset2 , cropZ1:cropZ2 );
                    fractureMask3_ = fractureMask3( centroid1(2)+cropOffset1:centroid1(2)+cropOffset2 , centroid1(1)+cropOffset1:centroid1(1)+cropOffset2 , cropZ1:cropZ2 );
                    boneMask3_     = boneMask3_ + uint8(fractureMask3_).*40;                                                                                                       % Output: Bone Mask3 + FractureMask3
                    %softInten3_     = softInten3(   centroid1(2)+cropOffset1:centroid1(2)+cropOffset2 , centroid1(1)+cropOffset1:centroid1(1)+cropOffset2 , cropZ1:cropZ2 );
                else
                    fprintf('   Skip augmentation of already fractured femur side  \n');
                end
                if fractureType(1) == 0       % For only intact right side !!!
                    % Simulate fracture
                    boneMask4 = uint8(pelvicMask40).*1+uint8(femurRMask40).*2;        % for simFracture2
                    [boneInten4,boneMask4,softInten4,softMask4,fractureMask4,frag4] = simFracture2(v40,boneMask4,softMask40,centroid2,cropZ1,k,angle);
                    airMask4 = boneMask4|softMask4|fractureMask4;
                    %fprintf('   NO. fragment4 = %u \n', frag4);
                    % Cropped volume >>> Output Intensity & Mask
                    boneInten4_    = boneInten4(    centroid2(2)+cropOffset1:centroid2(2)+cropOffset2 , centroid2(1)+cropOffset1:centroid2(1)+cropOffset2 , cropZ1:cropZ2 );       % Output: Intensity4
                    boneMask4_     = boneMask4(     centroid2(2)+cropOffset1:centroid2(2)+cropOffset2 , centroid2(1)+cropOffset1:centroid2(1)+cropOffset2 , cropZ1:cropZ2 );
                    fractureMask4_ = fractureMask4( centroid2(2)+cropOffset1:centroid2(2)+cropOffset2 , centroid2(1)+cropOffset1:centroid2(1)+cropOffset2 , cropZ1:cropZ2 );    
                    boneMask4_     = boneMask4_ + uint8(fractureMask4_).*40;                                                                                                       % Output: Bone Mask4 + FractureMask4
                    %softInten4_    = softInten4(    centroid2(2)+cropOffset1:centroid2(2)+cropOffset2 , centroid2(1)+cropOffset1:centroid2(1)+cropOffset2 , cropZ1:cropZ2 );
                else
                    fprintf('   Skip augmentation of already fractured femur side  \n');
                end
            end
            
            if first==1
                showOutputLogic = input('   Do you want to visual Target Output? [y/n] = ','s');
            end
            if showOutputLogic=='y'
                %figure; sliceViewer(boneMask3_);
                %figure; sliceViewer(boneMask4_);
                figure; sliceViewer(cat(2,boneInten3,softInten3)); title('boneInten3 + softInten3');
                figure; sliceViewer(cat(2,boneInten4,softInten4)); title('boneInten4 + softInten4');
                %figure; sliceViewer(softInten3);
                %figure; sliceViewer(softInten4);
                figure; set(gcf, 'Position', screensize.*[1 1 0.5 1]); volshow(boneMask3_,config3);
                %figure; set(gcf, 'Position', screensize.*[1 1 0.5 1]); volshow(boneInten3+softInten3,config2);
                figure; set(gcf, 'Position', [screensize(3)./2 0 screensize(3)./2 screensize(4)]); volshow(boneMask4_,config3);
                %figure; set(gcf, 'Position', [screensize(3)./2 0 screensize(3)./2 screensize(4)]); volshow(boneInten4+softInten4,config2);
            end
            
            clear boneMask3 boneMask4 
            clear pelvicMask3_ femurLMask3_ fractureMask3_ 
            clear pelvicMask4_ femurRMask4_ fractureMask4_ 
            
            
            
            %% Simulation of Radiograph for specific augmentNO
            
            % Input: [boneInten3-4, boneMask3-4, softInten3-4, airMask3-4]
            
            fprintf('   ### Simulation of Radiograph (pass Enter) ### \n')
            %if first==1
            %    pause()
            %end
            close all
            volumeViewer close

            for FemurSide = 1:2   % loop for left and right side
                if fractureType(mod(FemurSide,2)+1) == 4
                    fprintf('   Skip implanted femur side !!! \n');
                    continue;
                end
                if k > 0  &&  fractureType(mod(FemurSide,2)+1) > 0
                    fprintf('   Skip already fractured femur side from augmentation !!! \n');
                    continue;
                end
                fprintf('   #### Simulatation of Radiograph \n')
                if FemurSide==1
                    % Left-side
                    boneInten = boneInten3;
                    softInten = softInten3;
                    airMask = airMask3;
                    boneMask_ = boneMask3_;                                                 % >>> OUTPUT  (Cropped Bone Mask)
                    numFrag = frag3;                                                        % >>> OUTPUT  (Number of bone fragment)
                    centroid = centroid1;   % for croping
                else
                    % Rigth-side
                    boneInten = boneInten4;
                    softInten = softInten4;
                    airMask = airMask4;
                    boneMask_ = boneMask4_;                                                 % >>> OUTPUT  (Cropped Bone Mask)
                    numFrag = frag4;                                                        % >>> OUTPUT  (Number of bone fragment)
                    centroid = centroid2;   % for croping
                end
                
                % Simulate Radiograph
                if first == 1
                    u_water = input('   Input Attenuation Coefficient of Water [default=0.02mm^-1] = '); % at 70Kev  u_water=0.02 mm^-1  (but default = 0.005)
                end
                
                % Convert original intensity unit to Hounsfield unit
                ct = double(boneInten) + double(softInten);         % Intensities unit
                if b == 0
                    ctHU = ct - 1024;
                else
                    ctHU = ct.*double(m) + double(b);                                         % Hounsfield unit
                end
                clear ct boneInten softInten
                
                % Calculate Linear Attenuation Coefficient(LAC) map
                ctU = double(ctHU).*u_water./1000.0 + u_water;      % mm^-1
                clear ctHU
                
                crop_before = 2;
                %crop_before = input('   Mode [1] Crop before simDRR  [2] Crop after simDRR = ');
                if crop_before == 1
                    % Crop before
                    wtf
                    ctU = ctU( centroid(2)+cropOffset1:centroid1(2)+cropOffset2 , centroid(1)+cropOffset1:centroid(1)+cropOffset2 , cropZ1:cropZ2 );    % ผิดดดดดดดดด
                    sumX1 = squeeze(sum(ctU,1)).*sfactors2(1);
                    sumX2 = squeeze(sum(ctU,2)).*sfactors2(1);
                    sumX1 = imrotate(sumX1,90);
                    sumX2 = imrotate(sumX2,90);
                    drrImage1 = exp(-sumX1);        % int16((exp(-sumX1)).*4096);
                    drrImage2 = exp(-sumX2);        % int16((exp(-sumX2)).*4096);
                    drrImage1_ = drrImage1;         % >>> OUTPUT
                    drrImage2_ = drrImage2;         % >>> OUTPUT
                elseif crop_before == 2
                    % Crop after
                    sumX1 = squeeze(sum(ctU,1)).*sfactors2(1);
                    sumX2 = squeeze(sum(ctU,2)).*sfactors2(1);
                    sumX1 = imrotate(sumX1,90);
                    sumX2 = imrotate(sumX2,90);
                    drrImage1 = exp(-sumX1);        % int16((exp(-sumX1)).*4096);
                    drrImage2 = exp(-sumX2);        % int16((exp(-sumX2)).*4096);
                    drrImage1_ = drrImage1( size(v30,3)-cropZ2+1:size(v30,3)-cropZ1+1 , centroid(1)+cropOffset1:centroid(1)+cropOffset2 );        % >>> OUTPUT
                    drrImage2_ = drrImage2( size(v30,3)-cropZ2+1:size(v30,3)-cropZ1+1 , centroid(2)+cropOffset1:centroid(2)+cropOffset2 );        % >>> OUTPUT
                else
                    fprintf('   crop_before param. error !!! \n');
                    wtf
                end
                
                if first==1
                    showSimXLogic = input('   Do you want to visual Target Output with Simulated X-ray? [y/n] : ','s');
                end
                if showSimXLogic=='y'
                    % Show crop boundary
                    figure;  % fig = figure;
                    if FemurSide==1
                        set(gcf, 'Position', [0 0 screensize(3)./2 screensize(4)])   % gcf is current user interface
                    else
                        set(gcf, 'Position', [screensize(3)./2 0 screensize(3)./2 screensize(4)])
                    end
                    subplot(2,3,2); imshow(drrImage1,[0 0.75]); title('DRR with boundary (AP)'); hold on;
                    plot( centroid(1) , size(v30,3)-centroid(3) , 'r+', 'MarkerSize', 10, 'LineWidth', 2);
                    drawrectangle('Position',[centroid(1)+cropOffset1 size(v30,3)-cropZ2 256 256],'InteractionsAllowed','none');
                    subplot(2,3,5); imshow(drrImage2,[0 0.75]); title('DRR with boundary (LAT)'); hold on;
                    plot( centroid(2) , size(v30,3)-centroid(3) , 'r+', 'MarkerSize', 10, 'LineWidth', 2);
                    drawrectangle('Position',[centroid(2)+cropOffset1 size(v30,3)-cropZ2 256 256],'InteractionsAllowed','none');
                    
                    figure;
                    if FemurSide==1
                        set(gcf, 'Position', [screensize(3)./8 screensize(4)./4 screensize(3)./4 screensize(4)./2])         % gcf is current user interface
                    else
                        set(gcf, 'Position', [screensize(3).*5./8 screensize(4)./4 screensize(3)./4 screensize(4)./2])
                    end
                    
                end
                clear boneInten softInten airMask centroid
                clear ctU sumX1 sumX2 drrImage1 drrImage2
                
                %% Save Input and Target output
                
                fprintf('   #### Save {Input} and {Target Output} \n');
                saveLogic = input('   Save subject? [y/n] : ','s');
                if saveLogic == 'y'
                    sampleNO = file(1:end-4);
                    if first == 1   % only query for {Save Directory} at the first time
                        fileaddress = uigetdir('D:\FEW PhD\Datasets\Chula DICOM 2021\Cleaning Data\Dataset2','Specify Save Directory');     % On TYAN-GPU
                        first = 2;      % Don't ask for user input anymore
                        augmentSet = input('   Input augment set = ');
                    end
                    
                    % Assign data's file name
                    filename1 = strcat('V','-',sampleNO,'-',string(augmentSet),'-',string(FemurSide),'-',augmentNO,'.npy');  % filename1 = V-20200101CT0000-1-00.npy
                    filename2 = strcat('M','-',sampleNO,'-',string(augmentSet),'-',string(FemurSide),'-',augmentNO,'.npy');  % filename2 = M00011000.npy
                    filename3 = strcat('Y','-',sampleNO,'-',string(augmentSet),'-',string(FemurSide),'-',augmentNO,'.npy');  % filename3 = Y00011000.npy
                    filename4 = strcat('X','-',sampleNO,'-',string(augmentSet),'-',string(FemurSide),'-',augmentNO,'.npy');  % filename4 = X00011000.npy
                    
                    % Save DataSetLog.xlsx
                    if fractureType(mod(FemurSide,2)+1) == 0
                        if k == 0
                            fileaddress2 = strcat(fileaddress,'\','IntactLog',string(augmentSet),'.xlsx');
                        else
                            fileaddress2 = strcat(fileaddress,'\','AugmentLog',string(augmentSet),'.xlsx');
                        end
                    elseif fractureType(mod(FemurSide,2)+1) == 1
                        fileaddress2 = strcat(fileaddress,'\','NondisplaceLog',string(augmentSet),'.xlsx');
                    elseif fractureType(mod(FemurSide,2)+1) == 2
                        fileaddress2 = strcat(fileaddress,'\','DisplaceLog',string(augmentSet),'.xlsx');
                    else
                        wtf
                    end
                    
                    data = {filename1,filename2,filename3,filename4,numFrag};  % save dataset log in file .xlsx
                    writecell(data, fileaddress2, 'WriteMode', 'append');
                    %fprintf('   ### File address : %s   ###  \n', fileaddress);
                    %fprintf('   ### Files .xlsx is saved ###   \n')
                    
                    % Save .npy
                    filename1 = strcat(fileaddress,'\',filename1);  % filename1 = address...\V00011000.npy
                    filename2 = strcat(fileaddress,'\',filename2);  % filename2 = address...\M00011000.npy
                    filename3 = strcat(fileaddress,'\',filename3);  % filename3 = address...\Y00011000.npy
                    filename4 = strcat(fileaddress,'\',filename4);  % filename4 = address...\X00011000.npy
                    %writeNPY(v_, filename1);
                    writeNPY(uint8(boneMask_), filename2);
                    writeNPY(drrImage1_,filename3);
                    writeNPY(drrImage2_,filename4);
                    %fprintf('   ### Files Numpy is saved ### \n');
                    fprintf('   ### Finished sample = %s \n                       = %s \n                       = %s \n                       = %s \n',filename1,filename2,filename3,filename4);
                end
                
            end
            clear airMask3 airMask4
            pause(15);

        end
        clear v30 pelvicMask30 femurRMask30 femurLMask30 softMask30 fractureMask30
        clear v40 pelvicMask40 femurRMask40 femurLMask40 softMask40 fractureMask40
        
    end
    
    %% Finishing sample i and Clear variable to save RAM.
    
    timeEnd = toc(timeStart);
    sampleNO = file(1:end-4);
    fprintf('\n   Total sampling time for sample:%s = %f min. \n', sampleNO, timeEnd/60)
    
    if (pauseEachSample=='y')  % || (b == 0)
        fprintf('   Press [Enter] to go to the next sample : ');
        pause();
    end
    
    clear v2 femurLMask2 femurRMask2 pelvicMask2 softMask2 fractureMask2 
    clear info saveDiameter sfactors spatialDetails saveCentroid
    
    close all
    volumeViewer close
    imtool close all
    
end
fprintf('   Finish sampling @ Date = %u-%u-%u   Time = %u-%u-%f\n',clock)