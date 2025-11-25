# AI/ML Interview Preparation Tracker - Google

A personalized web application tailored to help you prepare for Google AI/ML roles with structured daily 2-hour study sessions over 60 days, **including coding practice, system design problems, and ML implementation challenges**.

## 🎯 Overview

This tracker is specifically designed based on your background as a **Cybersecurity AI Analyst** with extensive experience in MLOps, computer vision, NLP, and production ML systems. The 60-day plan covers everything you need to ace Google's AI/ML interview process.

## ✨ Features

- 📅 **60-Day Structured Plan**: Comprehensive daily 2-hour sessions covering ML fundamentals to advanced topics
- 💻 **Daily Coding Problems**: LeetCode-style problems organized by difficulty and topic (50+ days of problems!)
- 🏗️ **System Design Practice**: 16 real-world ML system design problems with detailed focus areas
- 🔨 **ML Coding Challenges**: Implement 14+ ML algorithms from scratch (NumPy only!)
- ✅ **Task Tracking**: Mark tasks as completed and carry over unfinished tasks
- 📚 **Learning Topics Reference**: Complete categorized list of AI/ML topics to master
- 📝 **Review Notes & Custom Topics**: Capture follow-ups and add new learning items
- 📊 **Progress Visualization**: Track completion percentage and maintain streaks
- 💾 **Local Storage**: Your progress is saved in your browser
- 🎯 **Google AI/ML Focus**: Tailored for Google's ML Engineer interview process

## 🚀 Getting Started

### Option 1: Open Locally (Easiest)

1. Navigate to the folder:
```bash
cd job-prep-tracker
```

2. Open `index.html` in your browser:
   - **macOS**: Double-click `index.html` or use Safari/Chrome to open it
   - **Command line**: `python3 -m http.server 8000` then visit `http://localhost:8000`

3. Start tracking your preparation!

### Option 2: Deploy to Netlify (Access Anywhere)

[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy)

#### Manual Deployment

1. Sign up at [Netlify](https://www.netlify.com/)
2. Drag and drop the entire `job-prep-tracker` folder to Netlify
3. Your app is live! 🎉

**Alternative: Netlify CLI**
```bash
npm install -g netlify-cli
cd job-prep-tracker
netlify deploy --prod
```

## ☁️ Cloud Sync (Netlify Functions + Postgres)

The tracker now supports an optional cloud backend so your **review notes** and **completed tasks** stay in sync across devices.

1. **Install dependencies**
   ```bash
   cd job-prep-tracker
   npm install
   ```
2. **Provision a Postgres database** (Supabase/Neon/Railway all have generous free tiers).
3. **Set Netlify environment variables** (either a single connection string or discrete values):
   ```
   POSTGRES_CONNECTION_STRING=postgres://user:pass@host:5432/dbname
   # or
   PGHOST=...
   PGPORT=5432
   PGUSER=...
   PGPASSWORD=...
   PGDATABASE=...
   POSTGRES_SSL=true   # set to false only if your provider disables SSL
   ```
4. **Deploy to Netlify**. The serverless functions live in `netlify/functions`:
   - `notes` → `/.netlify/functions/notes` (`GET`, `POST`, `DELETE`)
   - `tasks` → `/.netlify/functions/tasks` (`GET`, `POST`)

The functions automatically create the required tables:

```sql
CREATE TABLE IF NOT EXISTS review_notes (
  id SERIAL PRIMARY KEY,
  day INTEGER NOT NULL,
  note_text TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS completed_tasks (
  task_id TEXT PRIMARY KEY,
  completed BOOLEAN NOT NULL DEFAULT TRUE,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

If the API isn’t reachable (e.g., offline), the tracker gracefully falls back to localStorage so you never lose your daily plan.

## 📖 Usage

### First Time Setup

When you first open the app, you'll see a welcome screen that:
- ✅ Sets your **start date to today** (or a custom date you pick)
- ✅ Explains the 60-day journey ahead
- ✅ Shows your daily commitment (2 hours)
- ✅ Previews what's included

**Prefer a different start day?**
- Use the welcome screen countdown if you want to schedule your kickoff
- Preview content ahead of time and start whenever you're ready

### Main Dashboard

1. **Daily Study Plan**: View today's AI/ML learning tasks (typically 2-3 tasks totaling ~120 minutes)
2. **Progress Tracking**: See your overall completion %, completed tasks, and current streak
3. **Navigation**: Move between days to review past content or preview upcoming topics

### 💻 Coding Practice

Click **"Daily Coding Problems"** to view:
- **50+ days** of curated LeetCode-style problems
- Organized by week and topic (Arrays, Trees, DP, Graphs, etc.)
- Difficulty levels: Easy, Medium, Hard
- Direct links to LeetCode problems
- Topics covered: Hash Tables, Sliding Window, Binary Trees, Dynamic Programming, Backtracking, Graphs, Heaps, and more!

**Example Problems:**
- Day 1: Two Sum, Valid Parentheses, Best Time to Buy Stock
- Day 9: Number of Islands, Clone Graph, Course Schedule
- Day 40: Median of Two Sorted Arrays (Hard!)

### 🏗️ System Design Practice

Click **"System Design Problems"** to view:
- **16 comprehensive system design problems** tailored to ML roles
- Problems connected to YOUR experience (ADAS, security, MLOps)
- Detailed focus areas for each problem
- Estimated time to complete (45-60 min each)

**Example Problems:**
- **Day 14**: Netflix Recommendation System (relates to your ML pipelines)
- **Day 24**: Fraud Detection System (relates to your Beacon AI SAST work)
- **Day 32**: Google Photos Search (relates to your ADAS camera object detection!)
- **Day 40**: Voice Assistant Design (relates to your Vertex AI Gemini experience)
- **Day 44**: Autonomous Driving Perception (YOUR Continental ADAS expertise!)
- **Day 48**: Real-Time ML Model Serving (YOUR MLOps work!)

### 📚 Learning Topics Reference

Click **"View All Learning Topics"** to see:
- **12 categories** of AI/ML knowledge
- **100+ specific topics** to master
- Organized by: ML Fundamentals, Deep Learning, CV, NLP, RL, MLOps, System Design, Advanced Topics, Math, Tools, Data Engineering, Security & Ethics

### ➕ Add Custom Topics

Need to focus on something specific?
1. Click "+ Add Custom Topic"
2. Fill in: Name, Duration, Category, and Notes/Resources
3. The topic is added to your current day
4. Unfinished topics automatically carry over to the next day

### Daily Workflow

**First Time:**
1. Open the app → See welcome screen
2. Review your start date (tomorrow)
3. Click "Let's Get Started"
4. Preview Day 1 content (optional)

**Daily Routine:**
1. **Morning (or whenever you start):**
   - Open the tracker (auto-advances to today's day!)
   - Review today's tasks
   - Check AI insights and recommendations
   - Review carried-over tasks from yesterday

2. **During Study (2 hours):**
   - Work through coding problems (30-40 min)
   - Study the day's ML topic (60 min)
   - Optional: Practice system design if scheduled (40-50 min)

3. **End of Session:**
   - Check off completed tasks
   - AI model updates your insights
   - Review tomorrow's preview
   - Track your streak!

**Navigation:**
- **← Previous / Next →**: Browse any day
- **Jump to Today**: Instantly go to today's day
- **Date Badge**: Shows if day is Past/Today/Future
- **Auto-advance**: App automatically shows today's content each day

**Daily 2-Hour Session:**

### Core ML Topics
1. **Machine Learning Fundamentals** - Supervised, unsupervised learning, evaluation metrics
2. **Deep Learning** - Neural networks, CNNs, RNNs, Transformers
3. **Computer Vision** - Object detection, segmentation, 2D-to-3D mapping (leveraging your ADAS experience)
4. **Natural Language Processing** - LLMs, BERT, GPT, prompt engineering (your Vertex AI Gemini work)
5. **Reinforcement Learning** - MDP, Q-learning, policy gradients (your computational trust research)

### Production & Systems
6. **MLOps & Production ML** - Pipelines, monitoring, deployment (your Databricks & Hex experience)
7. **ML System Design** - Recommendation systems, search ranking, real-time ML
8. **Data Engineering for ML** - Feature stores, ETL, Snowflake (your current expertise)
9. **ML Security & Ethics** - Adversarial ML, fairness, AI governance (your cybersecurity background)

### Advanced Topics
10. **Graph Neural Networks** - GCN, GAT, message passing
11. **Generative AI** - GANs, VAEs, diffusion models
12. **Distributed ML** - Multi-GPU training, federated learning
13. **Model Optimization** - Quantization, pruning, edge deployment

### Interview Preparation
14. **ML Coding Interviews** - Implement algorithms from scratch
15. **ML Theory & Math** - Linear algebra, probability, optimization
16. **Behavioral Interviews** - STAR stories from your projects (Beacon AI, Databricks, ADAS)
17. **Google-Specific** - Google's ML infrastructure, research awareness

## 🚀 Getting Started

### Local Development

1. Clone this repository
```bash
cd job-prep-tracker
```

2. Open `index.html` in your browser
```bash
open index.html  # macOS
# or just double-click index.html
```

3. Start tracking your preparation!

### Deploy to Netlify

[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy)

#### Manual Deployment

1. Sign up at [Netlify](https://www.netlify.com/)
2. Drag and drop the entire `job-prep-tracker` folder to Netlify
3. Your app is live! 🎉

**Alternative: Netlify CLI**
```bash
npm install -g netlify-cli
cd job-prep-tracker
netlify deploy --prod
```

## 📖 Usage

### Daily Study Flow
1. **Open the Tracker**: Launch the app and check today's tasks
2. **Complete 2-Hour Session**: Work through the day's tasks (typically 2-3 tasks totaling ~120 minutes)
3. **Mark as Complete**: Check off finished tasks
4. **Carry Over**: Unfinished tasks automatically move to the next day

### Key Features

#### 📚 View Learning Topics
Click "View All Learning Topics" to see a comprehensive reference of all AI/ML concepts organized by category. Use this to:
- Understand the full scope of preparation
- Add missing topics to your custom learning plan
- Quick reference during study sessions

#### ➕ Add Custom Topics
Need to focus on something specific?
1. Click "+ Add Custom Topic"
2. Fill in: Name, Duration, Category, and Notes/Resources
3. The topic is added to your current day

#### 📊 Track Progress
- **Progress Bar**: Overall completion percentage
- **Stats**: Completed tasks, total tasks, current streak
- **Day Navigation**: Move between days to review or preview content

### Leveraging Your Experience

The plan is customized based on your resume:
- **Days 1-10**: Foundation & your strong areas (MLOps, CV, NLP)
- **Days 11-30**: Deep dives into system design and advanced ML
- **Days 31-45**: Google-specific prep and specializations
- **Days 46-60**: Interview polish, mocks, and confidence building

Your relevant projects to prepare STAR stories:
1. **Beacon AI SAST Platform** (Vertex AI, prompt orchestration, false positive reduction)
2. **Databricks ML Pipelines** (Security analytics, ML reviewer recommendations, anomaly detection)
3. **ADAS Camera Object Detection** (Computer vision, 2D-to-3D mapping, Kalman filters)
4. **Computational Trust Research** (IEEE publication, RL for robotics)
5. **Cloud-Native ML Services** (Django APIs, Docker/Kubernetes, CI/CD)

## 🛠️ Tech Stack

- **Pure HTML5, CSS3, and Vanilla JavaScript**
- No build process required
- No dependencies
- Fully static (can be hosted anywhere)
- LocalStorage for persistence

## 📁 Project Structure

```
job-prep-tracker/
├── index.html          # Main HTML structure
├── styles.css          # Styling and responsive design
├── app.js              # Application logic & state management
├── data.js             # 60-day preparation plan & learning topics
├── netlify.toml        # Netlify deployment config
└── README.md           # This file
```

## 🎯 Study Tips

1. **Consistency > Intensity**: 2 hours daily is better than 10 hours once a week
2. **Active Learning**: Implement algorithms, don't just read
3. **Connect to Experience**: Relate concepts to your Rivian, Bosch, Continental work
4. **Mock Interviews**: Critical - do at least 10 before the real interview
5. **STAR Stories**: Polish 15 stories covering all your achievements
6. **Google Research**: Read Google's ML papers and blog posts

## 🔄 Customization

### Modify the Study Plan
Edit `data.js` to:
- Adjust daily tasks
- Change time allocations
- Add/remove categories
- Update learning topics

### Change Styling
Edit `styles.css` to:
- Update color scheme (CSS variables in `:root`)
- Adjust layout and spacing
- Customize mobile responsiveness

## 📝 License

MIT License - Feel free to use and modify for your preparation!

## 🙏 Good Luck!

You have an impressive background spanning cybersecurity, MLOps, computer vision, and production ML systems. This 60-day plan will help you systematically prepare and showcase your expertise. Trust the process, stay consistent, and you'll be ready! 🚀

---

**Pro Tip**: Open this tracker daily, even if just for 5 minutes. Consistency builds momentum!
