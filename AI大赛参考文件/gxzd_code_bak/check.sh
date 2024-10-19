clear
echo '---------------------------------------------------检查预测代码---------------------------------------------'
ps -ef | grep predict.py
cat ./log/predict.log | grep 成功
cat ./log/predict.log | grep 失败
echo '-----------------------------------------------检查获取数据代码---------------------------------------------'
ps -ef | grep get_data.py
cat ./log/get_data.log | grep 成功
cat ./log/get_data.log | grep 失败
echo '-----------------------------------------------检查保底程序--------------------------------------------------'
ps -ef | grep baseline.py
cat ./log/baseline.log | grep 成功
cat ./log/get_data.log | grep 失败
