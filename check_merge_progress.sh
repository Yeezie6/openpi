#!/bin/bash
# 监控合并进度脚本

echo "正在监控合并进度..."
echo "日志文件: /tmp/merge_output.log"
echo "-----------------------------------"
echo ""

# 显示当前进度
tail -20 /tmp/merge_output.log | grep -E "处理数据集|找到|%|合并完成|统计信息|✅"

echo ""
echo "-----------------------------------"
echo "最新 10 行日志："
tail -10 /tmp/merge_output.log

echo ""
echo "进程状态："
ps aux | grep "merge_v2_direct.py" | grep -v grep || echo "进程已完成或未运行"

echo ""
echo "输出文件大小："
du -sh /pfs/pfs-ilWc5D/yiqing/openpi/lerobot/src/datasets/PickPlaceBottle_Merged 2>/dev/null || echo "输出目录不存在"
