
echo test all

/home/espnet/tools/venv/bin/python cal_bleu.py saved_models/multi_model/test_1.output saved_models/multi_model/test_1.gold

echo en-da
/home/espnet/tools/venv/bin/python cal_bleu.py saved_models/multi_model/test_1.en-da.da.output saved_models/multi_model/test_1.en-da.da.gold
echo da-en
/home/espnet/tools/venv/bin/python cal_bleu.py saved_models/multi_model/test_1.en-da.en.output saved_models/multi_model/test_1.en-da.en.gold

echo en-no
/home/espnet/tools/venv/bin/python cal_bleu.py saved_models/multi_model/test_1.en-no.no.output saved_models/multi_model/test_1.en-no.no.gold
echo no-en
/home/espnet/tools/venv/bin/python cal_bleu.py saved_models/multi_model/test_1.en-no.en.output saved_models/multi_model/test_1.en-no.en.gold

echo en-lv
/home/espnet/tools/venv/bin/python cal_bleu.py saved_models/multi_model/test_1.en-lv.lv.output saved_models/multi_model/test_1.en-lv.lv.gold
echo lv-en
/home/espnet/tools/venv/bin/python cal_bleu.py saved_models/multi_model/test_1.en-lv.en.output saved_models/multi_model/test_1.en-lv.en.gold

echo en-zh
/home/espnet/tools/venv/bin/python cal_bleu.py saved_models/multi_model/test_1.en-zh.zh.output saved_models/multi_model/test_1.en-zh.zh.gold
echo zh-en
/home/espnet/tools/venv/bin/python cal_bleu.py saved_models/multi_model/test_1.en-zh.en.output saved_models/multi_model/test_1.en-zh.en.gold

