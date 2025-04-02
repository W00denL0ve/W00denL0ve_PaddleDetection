@echo off

:: 初始化本地仓库
echo initiate...
git init
echo done

:: 配置用户信息
echo setting config...
git config --global user.name "W00denL0ve"
git config --global user.email "1317603153@qq.com"

:: 添加远程仓库（仓库地址）
echo adding remote git...
git remote rm origin
git remote add origin https://github.com/W00denL0ve/W00denL0ve_PaddleDetection.git
echo done

:: 添加所有文件到暂存区
echo add files to buffer...
git add .
echo done

:: 创建初始提交
echo create init...
git commit -m "Initial commit: Upload PaddleDetection project_2"
echo done

:: 推送到GitHub
echo pushing...
git push --force origin main
echo done

echo finish!
pause
