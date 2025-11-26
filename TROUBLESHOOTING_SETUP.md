# Troubleshooting: "Failed to complete setup" Error

## 🔍 What You're Seeing

Error message: "Failed to complete setup. Please try again."

This happens when the app can't save your setup data to the database.

---

## 🐛 Common Causes & Fixes

### 1. Netlify Functions Not Deployed Yet

**Problem:** The `preferences.js` function hasn't been deployed to Netlify yet.

**Check:**
- Open browser DevTools (F12)
- Go to Network tab
- Try submitting the form again
- Look for a request to `/.netlify/functions/preferences`
- If you see **404 Not Found** → Functions aren't deployed

**Fix:**
```bash
# Deploy the updated code
cd /Users/bhavananare/github/webapp/job-prep-tracker
git add .
git commit -m "Add user setup with preferences function"
git push origin main

# Wait 2-3 minutes for Netlify to rebuild
# Check Netlify dashboard: Site overview → Functions
# You should see: preferences, tasks, notes
```

---

### 2. Supabase Credentials Not Set

**Problem:** Environment variables are missing in Netlify.

**Check:**
1. Go to Netlify Dashboard
2. Your site → Site settings → Environment variables
3. Look for:
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY`

**Fix:**
```bash
# Add via Netlify Dashboard:
1. Site settings → Environment variables
2. Add variable:
   Key: SUPABASE_URL
   Value: https://xxxxxxxxxxxxx.supabase.co
   
3. Add variable:
   Key: SUPABASE_ANON_KEY
   Value: eyJhbGc...your-anon-key...xyz
   
4. Click "Save"
5. Go to Deploys → Trigger deploy → Deploy site
```

**Or via CLI:**
```bash
netlify env:set SUPABASE_URL "https://xxxxxxxxxxxxx.supabase.co"
netlify env:set SUPABASE_ANON_KEY "eyJhbGc...your-key...xyz"
netlify deploy --prod
```

---

### 3. Function Error (500)

**Problem:** Function is crashing due to missing dependency or code error.

**Check Netlify Function Logs:**
1. Netlify Dashboard → Functions
2. Click on `preferences` function
3. Check logs for errors

**Common Issues:**
- Missing `@supabase/supabase-js` package
- Syntax error in preferences.js
- Database connection failed

**Fix:**
```bash
# Ensure package.json has the dependency
cd /Users/bhavananare/github/webapp/job-prep-tracker
cat package.json
# Should include: "@supabase/supabase-js": "^2.39.0"

# If missing, add it:
npm install @supabase/supabase-js

# Commit and push
git add package.json package-lock.json
git commit -m "Add Supabase dependency"
git push origin main
```

---

### 4. Database Table Doesn't Exist

**Problem:** The `user_preferences` table hasn't been created in Supabase yet.

**Check:**
1. Go to https://app.supabase.com
2. Your project → Table Editor
3. Look for `user_preferences` table

**Fix:**
```bash
# If table doesn't exist, run the schema:
1. Go to Supabase → SQL Editor
2. Open local file: supabase-schema.sql
3. Copy entire contents
4. Paste into SQL Editor
5. Click "Run"
6. Check Table Editor → Should see 4 tables:
   - completed_tasks
   - review_notes
   - daily_progress
   - user_preferences
```

---

## 🚀 Quick Fix Steps

### Step 1: Check What's Deployed
```bash
# Check Netlify site
curl https://jobprepe.netlify.app/.netlify/functions/preferences
# Should NOT return 404
```

### Step 2: Verify Environment Variables
```bash
netlify env:list
# Should show:
# SUPABASE_URL
# SUPABASE_ANON_KEY
```

### Step 3: Check Function Logs
1. Netlify Dashboard → Functions → preferences
2. Look for recent errors
3. Note any error messages

### Step 4: Test Supabase Connection
```bash
# Create test file: test-supabase.js
const { createClient } = require('@supabase/supabase-js');

const supabaseUrl = 'YOUR_URL';
const supabaseKey = 'YOUR_KEY';
const supabase = createClient(supabaseUrl, supabaseKey);

async function test() {
  const { data, error } = await supabase
    .from('user_preferences')
    .select('*')
    .limit(1);
  
  console.log('Data:', data);
  console.log('Error:', error);
}

test();

# Run it:
node test-supabase.js
```

---

## 🔧 Temporary Workaround

While debugging, you can make the setup work without the database:

**Edit `app.js`** - Comment out the database save:

```javascript
async function handleUserSetup(form) {
    // ... existing code ...
    
    try {
        const userId = getUserId();
        
        // TEMPORARY: Comment out database save
        /*
        const response = await fetch(`${API_BASE}/preferences`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                userId,
                startDate,
                currentDay: 1,
                settings: { username, targetCompany, setupDate: new Date().toISOString() }
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to save user preferences');
        }
        */
        
        // Update state (works without database)
        state.username = username;
        state.startDate = startDate;
        state.targetCompany = targetCompany;
        state.userId = userId;
        state.currentDay = 1;
        
        // Save to localStorage (works without database)
        localStorage.setItem('userSetupComplete', 'true');
        saveState();
        
        // Rest of the code...
    }
}
```

This will make the app work locally, but data won't persist across browsers.

---

## 🎯 Most Likely Issue

Based on the screenshot, you just deployed the code. The most likely issues are:

1. **Functions not deployed yet** - Wait 2-3 minutes and try again
2. **Environment variables not set** - Add SUPABASE_URL and SUPABASE_ANON_KEY
3. **Database table not created** - Run supabase-schema.sql in Supabase

---

## ✅ Verification Checklist

After fixing, verify these work:

- [ ] Navigate to `https://jobprepe.netlify.app/.netlify/functions/preferences`
  - Should NOT return 404
  - Should return JSON (even if empty)

- [ ] Check Netlify Dashboard → Functions
  - [ ] `preferences` function exists
  - [ ] No errors in logs
  - [ ] Recent invocations show up

- [ ] Check Netlify Dashboard → Site settings → Environment variables
  - [ ] `SUPABASE_URL` is set
  - [ ] `SUPABASE_ANON_KEY` is set

- [ ] Check Supabase Dashboard → Table Editor
  - [ ] `user_preferences` table exists
  - [ ] Has correct columns (user_id, start_date, current_day, settings)

- [ ] Test the form
  - [ ] Fill in name + date
  - [ ] Click "Start My Journey"
  - [ ] Modal should close
  - [ ] Name appears in header
  - [ ] No error alert

---

## 📞 Need More Help?

If still not working, check:

1. **Browser Console** (F12 → Console tab)
   - Look for red errors
   - Copy exact error message

2. **Network Tab** (F12 → Network tab)
   - Filter by "fetch/xhr"
   - Click the failed request
   - Check Response tab
   - Send me the error details

3. **Netlify Function Logs**
   - Copy recent log entries
   - Look for stack traces

I can help debug further with this information! 🚀

