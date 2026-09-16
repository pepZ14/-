// 中青年人群肥胖风险预测（研究原型）—— 熵权法版本
// 模型：RandomForest（嵌套CV网格搜索调参，max_depth=10, min_samples_leaf=4）
// 特征：熵权法膳食模式得分(entropy_score) + 运动习惯 + 肥胖家族史 + EBQ总分 + 吸烟状态
// 全部计算在浏览器本地完成，不上传任何数据。

const GRP_DEFS = [
  { key: "grp_staple", label: "主食类", examples: "米饭、粥、包子/馒头、面条、米粉、糯米制品、方便面等" },
  { key: "grp_tuber", label: "薯类", examples: "土豆、红薯、芋头等" },
  { key: "grp_pickled_fried", label: "腌制/烧烤/油炸食品", examples: "腌菜、烟熏食品、烧烤、油炸食品" },
  { key: "grp_egg", label: "蛋类", examples: "鸡蛋、鸭蛋等" },
  { key: "grp_meat", label: "新鲜肉类", examples: "猪肉、牛肉、羊肉、禽肉、动物内脏等" },
  { key: "grp_seafood", label: "海鲜类", examples: "鱼、虾蟹、鱿鱼、海带紫菜等海产品" },
  { key: "grp_dairy", label: "奶制品", examples: "鲜奶、酸奶、奶粉、奶酪等" },
  { key: "grp_snack", label: "零食与坚果", examples: "薯片、糖果、坚果、西式糕点等" },
  { key: "grp_beverage", label: "饮料", examples: "含糖饮料、茶、咖啡" },
  { key: "grp_soy", label: "豆制品", examples: "豆腐、豆浆、豆干、腐竹等" },
  { key: "grp_vegetable", label: "新鲜蔬菜", examples: "叶菜、白菜甘蓝、十字花科、番茄、胡萝卜等各类蔬菜" },
  { key: "grp_fruit", label: "新鲜水果", examples: "苹果梨、香蕉、柑橘、瓜类、桃李、葡萄等各类水果" },
  { key: "grp_dried", label: "干制品", examples: "红枣、葡萄干、干香菇、海产干货等" },
];

const FREQ_OPTIONS = [
  { v: 1, t: "从不吃" },
  { v: 2, t: "偶尔吃" },
  { v: 3, t: "1-3次/月" },
  { v: 4, t: "1次/周" },
  { v: 5, t: "2-3次/周" },
  { v: 6, t: "4-6次/周" },
  { v: 7, t: "1次/天" },
  { v: 8, t: "2-3次/天" },
  { v: 9, t: "≥4次/天" },
];

// 熵权法得分严格公式是对78个原始FFQ题项做归一化后按各自信息熵权重加总，
// 为保持和其他两个网页一致的13输入体验，这里用训练数据对
// entropy_score ~ 13个食物大类均值得分 做OLS回归得到的近似线性公式（R²=0.980，
// 三个网页里精度最低的一个，但仍是很高精度的近似）。
const ENTROPY_INTERCEPT = -0.10068643449235687;
const ENTROPY_COEF = {
  grp_staple: 0.016046, grp_tuber: 0.003356, grp_pickled_fried: 0.016426, grp_egg: -0.000620,
  grp_meat: 0.003873, grp_seafood: 0.011317, grp_dairy: 0.007818, grp_snack: 0.007274,
  grp_beverage: 0.001551, grp_soy: 0.014872, grp_vegetable: 0.007865, grp_fruit: 0.017234,
  grp_dried: 0.017343,
};

const EBQ_ITEMS = [
  "您无法控制吃东西的欲望",
  "吃东西意味着您不用考虑消极的事情",
  "暴饮暴食是您自己可以做到的事情",
  "一旦您开始吃东西，就停不下来",
  "吃东西有助于控制您的情绪",
  "您认为应该享受像暴饮暴食一样的快乐",
  "您在对待食物方面没有意志力",
  "吃东西能让您的情绪保持在一个可以忍受的水平",
  "您认为暴饮暴食是好的体验",
  "您无法控制您的饮食",
  "吃东西帮助您应对消极的想法",
  "暴饮暴食让您可以为自己拥有一些美好的东西",
  "如果您不控制自己，您永远不会停止吃东西",
  "饮食帮助您应对负面情绪",
  "如果您多吃一点也不会有什么不同",
  "我没有办法停止吃东西",
  "吃东西是我应对不想要的感觉的最好方法",
  "我喜欢暴饮暴食",
];
const EBQ_OPTIONS = [
  { v: 1, t: "非常不同意" },
  { v: 2, t: "不同意" },
  { v: 3, t: "一般" },
  { v: 4, t: "同意" },
  { v: 5, t: "非常同意" },
];

let MODEL = null;
let MODEL_LOAD_ERROR = null;

async function loadModel() {
  try {
    const res = await fetch("model_entropy.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    MODEL = await res.json();
  } catch (err) {
    MODEL_LOAD_ERROR = err;
    console.error("模型加载失败：", err);
  }
}

function showLoadError() {
  const box = document.getElementById("result");
  box.hidden = false;
  const isFileProtocol = location.protocol === "file:";
  box.innerHTML = `
    <div class="result-card risk-high">
      <div class="result-text">模型加载失败，无法预测</div>
      <div class="result-sub">
        ${isFileProtocol
          ? "检测到您是直接双击打开的网页文件（地址栏以 file:// 开头）。出于浏览器安全限制，这种方式无法读取同目录下的 model_entropy.json 文件，因此点开始预测没有反应。<br/>解决方法：① 通过 GitHub Pages 访问部署好的网址（推荐，正式使用时用这种方式）；或 ② 本地测试时用命令行在该文件夹下运行 <code>python -m http.server 8000</code>，然后浏览器打开 http://localhost:8000 访问，不要直接双击html文件。"
          : "请检查网络连接或刷新页面重试；如果问题持续，可能是 model_entropy.json 文件缺失或损坏。"}
      </div>
    </div>`;
}

function buildForm() {
  const grpWrap = document.getElementById("grp-questions");
  GRP_DEFS.forEach((g, i) => {
    const div = document.createElement("div");
    div.className = "q-item";
    div.innerHTML = `
      <label class="q-label">${i + 1}. ${g.label}<span class="q-hint">例如：${g.examples}</span></label>
      <select name="${g.key}" required>
        <option value="" disabled selected>请选择您平常的食用频率</option>
        ${FREQ_OPTIONS.map(o => `<option value="${o.v}">${o.t}</option>`).join("")}
      </select>`;
    grpWrap.appendChild(div);
  });

  const ebqWrap = document.getElementById("ebq-questions");
  EBQ_ITEMS.forEach((text, i) => {
    const div = document.createElement("div");
    div.className = "q-item";
    div.innerHTML = `
      <label class="q-label">${i + 1}. ${text}</label>
      <div class="radio-row" data-name="ebq${i + 1}">
        ${EBQ_OPTIONS.map(o => `
          <label class="radio-opt">
            <input type="radio" name="ebq${i + 1}" value="${o.v}" required />
            <span>${o.t}</span>
          </label>`).join("")}
      </div>`;
    ebqWrap.appendChild(div);
  });
}

function computeEntropyScore(grpValues) {
  let s = ENTROPY_INTERCEPT;
  for (const g of GRP_DEFS) s += ENTROPY_COEF[g.key] * grpValues[g.key];
  return s;
}

function standardize(raw, mean, scale) {
  return (raw - mean) / scale;
}

function rfPredictProba(featureVector, model) {
  let sum = 0;
  for (const tree of model.trees) {
    let node = 0;
    const left = tree.children_left, right = tree.children_right;
    const feat = tree.feature, thr = tree.threshold;
    while (left[node] !== -1) {
      node = featureVector[feat[node]] <= thr[node] ? left[node] : right[node];
    }
    sum += tree.value1_ratio[node];
  }
  return sum / model.trees.length;
}

function predict(inputs) {
  const entropy_score = computeEntropyScore(inputs);
  const rawByCol = { entropy_score, exercise: inputs.exercise, fam_obesity: inputs.fam_obesity, ebq_total: inputs.ebq_total };

  const numVals = MODEL.num_cols.map((c, i) => standardize(rawByCol[c], MODEL.scaler_mean[i], MODEL.scaler_scale[i]));
  const smokingDummies = MODEL.smoking_categories.slice(1).map(cat => (inputs.smoking === cat ? 1 : 0));
  const x = [...numVals, ...smokingDummies];

  const proba = rfPredictProba(x, MODEL);
  const label = proba >= MODEL.best_threshold ? 1 : 0;
  return { entropy_score, proba, label };
}

function readForm(form) {
  const fd = new FormData(form);
  const grpValues = {};
  GRP_DEFS.forEach(g => { grpValues[g.key] = Number(fd.get(g.key)); });

  let ebqTotal = 0;
  for (let i = 1; i <= 18; i++) ebqTotal += Number(fd.get(`ebq${i}`));

  return {
    ...grpValues,
    exercise: Number(fd.get("exercise")),
    fam_obesity: Number(fd.get("fam_obesity")),
    smoking: Number(fd.get("smoking")),
    ebq_total: ebqTotal,
  };
}

function renderResult(res) {
  const box = document.getElementById("result");
  box.hidden = false;
  const pct = (res.proba * 100).toFixed(1);
  const riskClass = res.label === 1 ? "risk-high" : "risk-low";
  const riskText = res.label === 1 ? "较高风险（预测为肥胖组）" : "较低风险（预测为非肥胖组）";
  box.innerHTML = `
    <div class="result-card ${riskClass}">
      <div class="result-pct">${pct}%</div>
      <div class="result-text">模型预测肥胖概率 ｜ ${riskText}</div>
      <div class="result-sub">膳食模式得分：${res.entropy_score.toFixed(2)}</div>
    </div>`;
  box.scrollIntoView({ behavior: "smooth", block: "center" });
}

window.addEventListener("DOMContentLoaded", async () => {
  buildForm();
  await loadModel();
  if (MODEL_LOAD_ERROR) {
    showLoadError();
  }
  document.getElementById("predict-form").addEventListener("submit", (e) => {
    e.preventDefault();
    if (MODEL_LOAD_ERROR || !MODEL) {
      showLoadError();
      document.getElementById("result").scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }
    const form = e.target;
    if (!form.checkValidity()) {
      form.reportValidity();
      return;
    }
    try {
      const inputs = readForm(form);
      const res = predict(inputs);
      renderResult(res);
    } catch (err) {
      console.error("预测出错：", err);
      const box = document.getElementById("result");
      box.hidden = false;
      box.innerHTML = `<div class="result-card risk-high"><div class="result-text">预测出错</div><div class="result-sub">${err.message || err}</div></div>`;
      box.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  });
});
