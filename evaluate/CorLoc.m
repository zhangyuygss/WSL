clc;
clear;
testset = 'trainval';
roundNum = 'round4';
rootPath = '/home/ggy/disk1/ggy/code/SSL_WSDDN';
datasetPath = fullfile(rootPath, 'data','VOC2007');
dataPath = fullfile(rootPath,'data','ssl_800_eb_ssw_ps');
resPath = fullfile(dataPath,roundNum,'0.1','FastRcnn','round4_pr_8','EB','results');
corlocPath = fullfile(dataPath,roundNum,'0.1','FastRcnn','round4_pr_8','EB',['corloc' testset '.txt']);
detsRes = load(fullfile(dataPath,roundNum,'0.1','FastRcnn','round4_pr_8','EB','detsTrainval.mat'));
addpath(fullfile(datasetPath,'VOCdevkit', 'VOCcode'));

if ~exist(resPath)
	mkdir(resPath);
end
imdb = load(fullfile(dataPath,'imdbwsddn.mat'));

imdb.images.set(imdb.images.set == 2) = 1;
trainIdx = find( imdb.images.set == 1);



VOCinit;
VOCdevkitPath = fullfile(datasetPath,'VOCdevkit');

VOCopts.imgsetpath = fullfile(VOCdevkitPath,'VOC2007','ImageSets','Main','%s.txt');
VOCopts.annopath   = fullfile(VOCdevkitPath,'VOC2007','Annotations','%s.xml');
VOCopts.localdir   = fullfile(VOCdevkitPath,'local','VOC2007');


resPath = fullfile(resPath,['%s_det_' testset '_%s.txt']);
boxscores_nms = detsRes .boxscores_nms;
%write det results to txt files
%chose only one box for one class of every image
for c=1:numel(VOCopts.classes)
    fid = fopen(sprintf(resPath,'comp4',VOCopts.classes{c}),'w');
    for i=1:numel(trainIdx)
        if isempty(boxscores_nms{c,i}), continue; end
        dets = boxscores_nms{c,i};
%         for j=1:size(dets,1)
%             fprintf(fid,'%s %.6f %d %d %d %d\n', ...
%                 imdb.images.name{trainIdx(i)}(1:end-4), ...
%                 dets(j,5),dets(j,1:4)) ;
%         end
        [~,j]=max(dets(:,5));
        fprintf(fid,'%s %.6f %d %d %d %d\n', ...
            imdb.images.name{trainIdx(i)}(1:end-4), ...
            dets(j,5),dets(j,1:4)) ;

    end
    fclose(fid);
end
    
corloc = zeros(1, VOCopts.nclasses);
for c = 1:VOCopts.nclasses
    cls = VOCopts.classes{c};
	
	
	% compute and display PR

    [gtids,t]=textread(sprintf(VOCopts.imgsetpath,testset),'%s %d');
    for i=1:length(gtids)
        % display progress
        tic;
        if toc>1
            fprintf('%s: pr: load: %d/%d\n',cls,i,length(gtids));
            drawnow;
            tic;
        end
        
        % read annotation
        recs(i)=PASreadrecord(sprintf(VOCopts.annopath,gtids{i}));
    end
	
	fprintf('%s: pr: evaluating discovery\n',cls);

    
    nimg=0;
    gt(length(gtids))=struct('BB',[],'diff',[],'det',[]);
    for i=1:length(gtids)
        % extract objects of class
        clsinds=strmatch(cls,{recs(i).objects(:).class},'exact');
        gt(i).BB=cat(1,recs(i).objects(clsinds).bbox)';
        gt(i).diff=[recs(i).objects(clsinds).difficult];
        gt(i).det=false(length(clsinds),1);
        nimg=nimg+double(size(gt(i).BB, 2)>0);
    end
    
    % load results
    [ids,scores,b1,b2,b3,b4]=textread(sprintf(resPath,'comp4',cls),'%s %f %f %f %f %f');
    BB=[b1 b2 b3 b4]' + 1;
    
    nd = length(ids);
    tp=zeros(nd,1);
    for d = 1:nd
        i=strmatch(ids{d},gtids,'exact');
        if isempty(i)
            error('unrecognized image "%s"',ids{d});
        elseif length(i)>1
            error('multiple image "%s"',ids{d});
        end
        
        bb=BB(:,d);
        ovmax=-inf;
        for j=1:size(gt(i).BB,2)
            bbgt=gt(i).BB(:,j);
            bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
            iw=bi(3)-bi(1)+1;
            ih=bi(4)-bi(2)+1;
            if iw>0 && ih>0
                % compute overlap as area of intersection / area of union
                ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
                    (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
                    iw*ih;
                ov=iw*ih/ua;
                if ov>ovmax
                    ovmax=ov;
                    jmax=j;
                end
            end
        end
        % assign detection as true positive/don't care/false positive
        if ovmax>=VOCopts.minoverlap
            %if ~gt(i).diff(jmax)
            %if ~gt(i).det(jmax)
            tp(d)=1;            % true positive
            % false positive (multiple detection)
            %end
            %end
        end
    end
    
    corloc(c) = sum(tp) / nimg;
    
    fprintf('corloc for class %s is %f.\n', cls, corloc(c) );
    
end

mcorloc = mean(corloc);
%write result to txt file
fid = fopen(corlocPath,'wt');
fprintf(fid,'%s%f\n','mcorloc: ',mcorloc);
fprintf(fid,'%f\n',corloc);
fclose(fid)