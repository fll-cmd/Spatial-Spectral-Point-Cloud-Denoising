clc; clear; close all;

%% ================= 配置 =================
gt_dir = 'E:\MATLAB程序\1.3\1.3.2\TESTPC4';
denoise_dir = 'E:\MATLAB程序\1.3\1.3.2\TESTPC4\消融实验';

excel_out = fullfile(denoise_dir, 'TESTPC4_消融实验评价结果.xlsx');

dist_thresh = 0.01;   % KDTree 匹配距离阈值
lambda = 5.0;         % ScoreProduct 权重

save_fig = false;
show_fig = false;
fig_dir = denoise_dir;

%% ================= 文件列表 =================
file_names = {};
for group_id = 1:3
    for sample_id = 1:10
        file_names{end+1,1} = sprintf('%d-%d', group_id, sample_id);
    end
end

ablation_names = {
    '1_A_only'
    '2_B_only'
    '3_C_only'
    '4_BC_noA'
    '5_AC_noB'
    '6_AB_noC'
    '7_Full_ABC'
};

%% ================= 结果表初始化 =================
nFiles = numel(file_names);
nAblations = numel(ablation_names);
nRows = nFiles * nAblations;

results_table = table( ...
    strings(nRows,1), strings(nRows,1), strings(nRows,1), ...
    zeros(nRows,1), zeros(nRows,1), ...
    zeros(nRows,1), zeros(nRows,1), zeros(nRows,1), zeros(nRows,1), ...
    zeros(nRows,1), zeros(nRows,1), zeros(nRows,1), ...
    zeros(nRows,1), zeros(nRows,1), zeros(nRows,1), ...
    'VariableNames', { ...
        'FileName','AblationName','DenoiseFile', ...
        'GTPoints','DenoisedPoints', ...
        'TP','FP','TN','FN', ...
        'NoiseRecall','Overkill','OneMinusOverkill', ...
        'Precision','ScoreLinear','ScoreProduct'} ...
);

%% ================= 批量评价 =================
row = 0;

for i = 1:nFiles

    file_name = file_names{i};
    gt_file = fullfile(gt_dir, [file_name, '.txt']);

    if ~isfile(gt_file)
        fprintf('原始文件不存在，跳过: %s\n', gt_file);
        continue;
    end

    for j = 1:nAblations

        ablation_name = ablation_names{j};
        denoise_file = fullfile(denoise_dir, ...
            [file_name, '_', ablation_name, '.txt']);

        if ~isfile(denoise_file)
            fprintf('去噪文件不存在，跳过: %s\n', denoise_file);
            continue;
        end

        row = row + 1;

        method_name = [file_name, '_', ablation_name];

        fprintf('\n==============================\n');
        fprintf('原始文件: %s\n', gt_file);
        fprintf('去噪文件: %s\n', denoise_file);

        m = denoise_evaluation( ...
            gt_file, denoise_file, dist_thresh, lambda, ...
            method_name, fig_dir, save_fig, show_fig);

        results_table.FileName(row) = string(file_name);
        results_table.AblationName(row) = string(ablation_name);
        results_table.DenoiseFile(row) = string(denoise_file);

        results_table.GTPoints(row) = m.GTPoints;
        results_table.DenoisedPoints(row) = m.DenoisedPoints;

        results_table.TP(row) = m.TP;
        results_table.FP(row) = m.FP;
        results_table.TN(row) = m.TN;
        results_table.FN(row) = m.FN;

        results_table.NoiseRecall(row) = m.NoiseRecall;
        results_table.Overkill(row) = m.Overkill;
        results_table.OneMinusOverkill(row) = 1 - m.Overkill;
        results_table.Precision(row) = m.Precision;
        results_table.ScoreLinear(row) = m.Score;
        results_table.ScoreProduct(row) = m.ScoreProduct;
    end
end

%% ================= 删除空行并写入 Excel =================
results_table = results_table(1:row, :);

writetable(results_table, excel_out, 'Sheet', 'Results');

fprintf('\n====================================\n');
fprintf('批量评价完成。\n');
fprintf('有效评价数量: %d\n', row);
fprintf('结果已保存到:\n%s\n', excel_out);

%% ================= 按消融方法统计均值和标准差 =================
if row > 0
    summary_table = groupsummary( ...
        results_table, ...
        'AblationName', ...
        {'mean','std'}, ...
        {'NoiseRecall','Overkill','OneMinusOverkill','Precision','ScoreLinear','ScoreProduct'} ...
    );

    writetable(summary_table, excel_out, 'Sheet', 'Summary');

    fprintf('统计汇总已写入 Sheet: Summary\n');
end

%% ================= 生成论文表格格式 Excel =================

paper_excel_out = fullfile(gt_dir, 'TESTPC4_消融实验_论文表格格式.xlsx');

% 方法名称映射
method_order = {
    '2_B_only',       'B邻域距离特征'
    '3_C_only',       'C法向一致性特征'
    '1_A_only',       'A频域异常度特征'
    '4_BC_noA',       'B邻域距离+C法向一致性特征'
    '5_AC_noB',       'A频域异常度+C法向一致性特征'
    '6_AB_noC',       'A频域异常度+B邻域距离特征'
    '7_Full_ABC',     'A+B+C多维异常度特征'
};

metric_names = {
    'OneMinusOverkill', '1-Overkill'
    'NoiseRecall',     'Noise Recall'
    'ScoreProduct',    'Score'
};

% 初始化表格内容
out_cell = cell(1 + 3 * 7, 6);

out_cell(1,:) = {
    '指标', ...
    '去噪方法', ...
    'PC1-1（mean±std）', ...
    'PC2-1（mean±std）', ...
    'PC3-1（mean±std）', ...
    'PC4-1（mean±std）'
};

row_id = 2;

for m = 1:size(metric_names,1)

    metric_field = metric_names{m,1};
    metric_label = metric_names{m,2};

    for k = 1:size(method_order,1)

        ablation_key = method_order{k,1};
        method_label = method_order{k,2};

        idx = results_table.AblationName == string(ablation_key);

        values = results_table.(metric_field)(idx);

        values = values(~isnan(values));

        if isempty(values)
            value_str = '';
        else
            value_mean = mean(values);
            value_std  = std(values);
            value_str = sprintf('%.4f ± %.4f', value_mean, value_std);
        end

        if k == 1
            out_cell{row_id,1} = metric_label;
        else
            out_cell{row_id,1} = '';
        end

        out_cell{row_id,2} = method_label;

        % 目前是 TESTPC4，所以只写 PC4 列
        out_cell{row_id,3} = '';
        out_cell{row_id,4} = '';
        out_cell{row_id,5} = '';
        out_cell{row_id,6} = value_str;

        row_id = row_id + 1;
    end
end

writecell(out_cell, paper_excel_out, 'Sheet', 'PaperTable');

fprintf('\n论文表格格式 Excel 已保存到:\n%s\n', paper_excel_out);
%% ================= 函数定义 =================
function metrics = denoise_evaluation(gt_file, denoise_file, dist_thresh, lambda, name, base_dir, save_fig, show_fig)

    %% 读取数据
    gt = readmatrix(gt_file);
    denoise = readmatrix(denoise_file);

    gt = gt(~any(isnan(gt),2), :);
    denoise = denoise(~any(isnan(denoise),2), :);

    if size(gt,2) < 4
        error('GT 文件必须至少包含 4 列: x y z label。文件: %s', gt_file);
    end

    if size(denoise,2) < 3
        error('去噪文件必须至少包含 3 列: x y z。文件: %s', denoise_file);
    end

    gt_xyz = gt(:,1:3);
    gt_label = gt(:,4);

    denoise_xyz = denoise(:,1:3);

    %% KDTree 匹配
    tree = KDTreeSearcher(denoise_xyz);
    [~, dist] = knnsearch(tree, gt_xyz);

    kept = dist < dist_thresh;

    %% 标签定义
    noise = gt_label == 1;
    clean = gt_label == 0;

    %% 混淆矩阵
    TP_idx = noise & ~kept;  % 噪声被删除
    FP_idx = clean & ~kept;  % 干净点被误删
    TN_idx = clean & kept;   % 干净点被保留
    FN_idx = noise & kept;   % 噪声被保留

    TP = sum(TP_idx);
    FP = sum(FP_idx);
    TN = sum(TN_idx);
    FN = sum(FN_idx);

    %% 指标计算
    NoiseRecall = TP / (TP + FN + eps);
    Overkill = FP / (FP + TN + eps);
    OneMinusOverkill = 1 - Overkill;
    Precision = TP / (TP + FP + eps);

    Score = NoiseRecall - lambda * Overkill;
    ScoreProduct = NoiseRecall * (1 - Overkill)^lambda;

    %% 输出
    fprintf('\n========== 去噪评估: %s ==========\n', name);
    fprintf('GT总点数: %d | 去噪后点数: %d\n', size(gt,1), size(denoise,1));
    fprintf('TP=%d FP=%d TN=%d FN=%d\n', TP, FP, TN, FN);
    fprintf('Noise Recall     = %.6f\n', NoiseRecall);
    fprintf('Overkill         = %.6f\n', Overkill);
    fprintf('1 - Overkill     = %.6f\n', OneMinusOverkill);
    fprintf('Precision        = %.6f\n', Precision);
    fprintf('Score Linear     = %.6f\n', Score);
    fprintf('Score Product    = %.6f\n', ScoreProduct);

    %% 可视化
    if show_fig || save_fig
        fig = figure('Color','w');

        if ~show_fig
            set(fig, 'Visible', 'off');
        end

        hold on;

        scatter3(gt_xyz(TN_idx,1), gt_xyz(TN_idx,2), gt_xyz(TN_idx,3), ...
            5, 'b', 'filled');

        scatter3(gt_xyz(noise,1), gt_xyz(noise,2), gt_xyz(noise,3), ...
            10, 'r', 'filled');

        scatter3(gt_xyz(FP_idx,1), gt_xyz(FP_idx,2), gt_xyz(FP_idx,3), ...
            18, 'k', 'filled');

        axis equal;
        grid on;
        view(3);

        xlabel('X');
        ylabel('Y');
        zlabel('Z');

        title(sprintf('%s FP 可视化', name), 'Interpreter','none');

        legend( ...
            'TN: 干净保留', ...
            '真实噪声', ...
            'FP: 干净被删', ...
            'Location','best');

        if save_fig
            fig_file = fullfile(base_dir, [name, '_FP_clean_deleted.png']);
            exportgraphics(fig, fig_file, 'Resolution', 300);
            fprintf('图片已保存: %s\n', fig_file);
        end

        if ~show_fig
            close(fig);
        end
    end

    %% 返回结果
    metrics.GTPoints = size(gt,1);
    metrics.DenoisedPoints = size(denoise,1);

    metrics.TP = TP;
    metrics.FP = FP;
    metrics.TN = TN;
    metrics.FN = FN;

    metrics.NoiseRecall = NoiseRecall;
    metrics.Overkill = Overkill;
    metrics.Precision = Precision;

    metrics.Score = Score;
    metrics.ScoreProduct = ScoreProduct;
end