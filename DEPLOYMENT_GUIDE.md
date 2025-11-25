# Complete Deployment Guide: Job Prep Tracker with Supabase + Netlify

## 📋 Overview

This guide shows you exactly how to deploy your Job Prep Tracker to Netlify with Supabase database.

**Important:** You do NOT need a `.env` file in Netlify. Environment variables are set through the Netlify dashboard.

---

## 🗂️ File Structure

```
job-prep-tracker/
├── index.html              ← Frontend
├── styles.css              ← Styles
├── app.js                  ← Frontend logic
├── data.js                 ← Daily tasks data
├── practice_problems.yaml  ← Practice problems
├── netlify/
│   └── functions/
│       ├── supabase-client.js  ← Database connection
│       ├── tasks.js            ← Tasks API
│       └── notes.js            ← Notes API
├── netlify.toml            ← Netlify configuration
├── package.json            ← Dependencies
├── supabase-schema.sql     ← Database schema
├── .env                    ← LOCAL ONLY (for testing)
└── .gitignore              ← Must include .env
```

---

## 🚀 Step-by-Step Deployment

### Step 1: Set Up Supabase Database

1. **Go to Supabase:**
   - Visit: https://app.supabase.com
   - Sign in with GitHub

2. **Create New Project:**
   - Click "New Project"
   - Name: `job-prep-tracker`
   - Database Password: Generate a strong password (save it!)
   - Region: Choose closest to you (e.g., US East)
   - Click "Create new project"
   - Wait 2-3 minutes for setup

3. **Run SQL Schema:**
   - In left sidebar: Click **SQL Editor**
   - Click **New query**
   - Open your local file: `supabase-schema.sql`
   - Copy entire contents
   - Paste into SQL Editor
   - Click **Run** (bottom right)
   - You should see: "Success. No rows returned"

4. **Get Your Credentials:**
   - In left sidebar: Click **Project Settings** (gear icon)
   - Click **API** tab
   - Copy these two values (you'll need them for Netlify):
     ```
     Project URL: https://xxxxxxxxxxxxx.supabase.co
     anon public key: eyJhbGc...very-long-token...xyz
     ```

---

### Step 2: Prepare Your Code

1. **Check package.json:**
   ```bash
   cd /Users/bhavananare/github/webapp/job-prep-tracker
   cat package.json
   ```

   Make sure it has:
   ```json
   {
     "name": "job-prep-tracker",
     "version": "1.0.0",
     "type": "module",
     "dependencies": {
       "@supabase/supabase-js": "^2.39.0"
     }
   }
   ```

2. **Check .gitignore:**
   ```bash
   cat .gitignore
   ```

   Make sure it includes:
   ```
   .env
   node_modules/
   .netlify/
   ```

3. **Commit Your Code:**
   ```bash
   git add .
   git commit -m "Add Supabase database integration"
   git push origin main
   ```

---

### Step 3: Deploy to Netlify

#### Option A: Deploy via Netlify Dashboard (Easiest)

1. **Go to Netlify:**
   - Visit: https://app.netlify.com
   - Sign in with GitHub

2. **Add New Site:**
   - Click "Add new site" → "Import an existing project"
   - Click "Deploy with GitHub"
   - Authorize Netlify if prompted
   - Select your repository: `webapp`

3. **Configure Build Settings:**
   - **Base directory:** `job-prep-tracker`
   - **Build command:** Leave empty (or `npm install`)
   - **Publish directory:** `.` (dot means current directory)
   - **Functions directory:** `netlify/functions`

4. **Add Environment Variables (CRITICAL!):**
   - Scroll down to "Environment variables"
   - Click "Add environment variables"
   - Add these two variables:

   **Variable 1:**
   ```
   Key: SUPABASE_URL
   Value: https://xxxxxxxxxxxxx.supabase.co
   (paste your Supabase Project URL from Step 1.4)
   ```

   **Variable 2:**
   ```
   Key: SUPABASE_ANON_KEY
   Value: eyJhbGc...your-long-anon-key...xyz
   (paste your Supabase anon public key from Step 1.4)
   ```

5. **Deploy:**
   - Click "Deploy webapp" (or similar button)
   - Wait 2-3 minutes
   - Your site will be live at: `https://random-name-123.netlify.app`

#### Option B: Deploy via Netlify CLI

```bash
# Install Netlify CLI (if not installed)
npm install -g netlify-cli

# Login to Netlify
netlify login

# Initialize (from job-prep-tracker directory)
cd /Users/bhavananare/github/webapp/job-prep-tracker
netlify init

# Follow prompts:
# - Create & configure a new site
# - Team: Your team
# - Site name: job-prep-tracker (or custom)
# - Build command: (leave empty or npm install)
# - Directory to deploy: .
# - Netlify functions folder: netlify/functions

# Add environment variables
netlify env:set SUPABASE_URL "https://xxxxxxxxxxxxx.supabase.co"
netlify env:set SUPABASE_ANON_KEY "eyJhbGc...your-key...xyz"

# Deploy
netlify deploy --prod
```

---

### Step 4: Verify Deployment

1. **Open Your Site:**
   - Visit your Netlify URL (e.g., `https://your-site.netlify.app`)

2. **Test the App:**
   - Check if daily tasks load
   - Try marking a task as complete
   - Add a review note
   - Refresh the page - data should persist!

3. **Check Netlify Functions:**
   - In Netlify Dashboard: Go to "Functions" tab
   - You should see:
     - `tasks`
     - `notes`
   - Both should show recent invocations

4. **Verify Database:**
   - Go back to Supabase dashboard
   - Click "Table Editor" in left sidebar
   - Select `completed_tasks` table
   - You should see your test data!

---

### Step 5: (Optional) Custom Domain

1. **In Netlify Dashboard:**
   - Go to "Site settings" → "Domain management"
   - Click "Add custom domain"
   - Enter your domain (e.g., `jobprep.yourdomain.com`)
   - Follow DNS configuration instructions

2. **Enable HTTPS:**
   - Netlify automatically provisions SSL certificate
   - Wait a few minutes for DNS propagation

---

## 🔧 Local Development Setup

For testing locally before deploying:

1. **Create `.env` file:**
   ```bash
   cd /Users/bhavananare/github/webapp/job-prep-tracker
   nano .env
   ```

2. **Add your Supabase credentials:**
   ```env
   SUPABASE_URL=https://xxxxxxxxxxxxx.supabase.co
   SUPABASE_ANON_KEY=eyJhbGc...your-key...xyz
   ```

3. **Install dependencies:**
   ```bash
   npm install
   ```

4. **Run locally:**
   ```bash
   netlify dev
   ```

5. **Test:**
   - Open: http://localhost:8888
   - Test all features

**Note:** `.env` file is ONLY for local testing. It's in `.gitignore` and won't be committed to Git.

---

## 📍 Where Does .env Go?

### For Local Development:
```
/Users/bhavananare/github/webapp/job-prep-tracker/.env
```
(Same directory as package.json)

### For Netlify Deployment:
**NO .env file needed!** Environment variables go in:
- **Netlify Dashboard** → Your Site → **Site settings** → **Environment variables**

OR

- Via CLI: `netlify env:set KEY "value"`

---

## 🐛 Troubleshooting

### Functions Not Working?

1. **Check Netlify Functions Logs:**
   - Dashboard → Functions → Click function name → View logs

2. **Check Environment Variables:**
   - Dashboard → Site settings → Environment variables
   - Verify `SUPABASE_URL` and `SUPABASE_ANON_KEY` are set

3. **Redeploy:**
   - Dashboard → Deploys → Trigger deploy → Deploy site

### Database Connection Issues?

1. **Check Supabase is Running:**
   - Go to https://app.supabase.com
   - Check project status (green = healthy)

2. **Verify Schema:**
   - Table Editor → Check tables exist:
     - `completed_tasks`
     - `review_notes`
     - `daily_progress`
     - `user_preferences`

3. **Check RLS Policies:**
   - Authentication → Policies
   - Ensure tables have "Allow all" policies (for now)

### Data Not Persisting?

1. **Open Browser Console (F12)**
   - Look for errors
   - Check Network tab for failed API calls

2. **Test API Endpoints:**
   ```bash
   # Test tasks endpoint
   curl https://your-site.netlify.app/.netlify/functions/tasks
   
   # Test notes endpoint
   curl https://your-site.netlify.app/.netlify/functions/notes
   ```

---

## ✅ Deployment Checklist

- [ ] Supabase project created
- [ ] Database schema run successfully
- [ ] Supabase credentials copied
- [ ] Code committed and pushed to GitHub
- [ ] Netlify site created from GitHub repo
- [ ] Base directory set to `job-prep-tracker`
- [ ] Environment variables added in Netlify:
  - [ ] `SUPABASE_URL`
  - [ ] `SUPABASE_ANON_KEY`
- [ ] Site deployed successfully
- [ ] Functions showing in Netlify dashboard
- [ ] Tested marking tasks complete
- [ ] Tested adding review notes
- [ ] Data persists after page refresh

---

## 📊 Summary

**What you need to do:**

1. ✅ Set up Supabase (done - you already have it)
2. ✅ Run SQL schema in Supabase SQL Editor
3. 📍 Push code to GitHub
4. 🚀 Deploy to Netlify from GitHub
5. 🔐 Add environment variables in Netlify Dashboard
6. ✅ Test the deployed site

**Where does .env go?**
- **Local testing:** `/Users/bhavananare/github/webapp/job-prep-tracker/.env`
- **Netlify deployment:** NO .env file - use Netlify Dashboard environment variables

**Your credentials from Supabase:**
- Get from: Project Settings → API tab
- Add to: Netlify Site Settings → Environment variables

---

## 🎉 You're Done!

Once deployed, your job prep tracker will:
- ✅ Track completed tasks in Supabase
- ✅ Store review notes permanently
- ✅ Sync across all devices
- ✅ Work offline with localStorage fallback
- ✅ Auto-deploy on every git push

Need help? Check the troubleshooting section above!

