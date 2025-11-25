-- Job Prep Tracker Database Schema for Supabase
-- Run this in Supabase SQL Editor: https://app.supabase.com

-- Table: completed_tasks
-- Stores which tasks the user has completed
CREATE TABLE IF NOT EXISTS completed_tasks (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) DEFAULT 'default_user',
    task_id VARCHAR(255) NOT NULL,
    day INTEGER NOT NULL,
    category VARCHAR(100) NOT NULL,
    task_title TEXT,
    completed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_user_task UNIQUE(user_id, task_id)
);

-- Table: review_notes
-- Stores notes for tasks that user wants to review later
CREATE TABLE IF NOT EXISTS review_notes (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) DEFAULT 'default_user',
    task_id VARCHAR(255) NOT NULL,
    day INTEGER NOT NULL,
    category VARCHAR(100) NOT NULL,
    task_title TEXT,
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table: daily_progress
-- Tracks daily progress statistics
CREATE TABLE IF NOT EXISTS daily_progress (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) DEFAULT 'default_user',
    date DATE NOT NULL,
    tasks_completed INTEGER DEFAULT 0,
    total_tasks INTEGER DEFAULT 0,
    categories_completed JSONB DEFAULT '{}',
    streak_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_user_date UNIQUE(user_id, date)
);

-- Table: user_preferences
-- Stores user settings and preferences
CREATE TABLE IF NOT EXISTS user_preferences (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) DEFAULT 'default_user',
    start_date DATE NOT NULL,
    current_day INTEGER DEFAULT 1,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_user_prefs UNIQUE(user_id)
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_completed_tasks_user_id ON completed_tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_completed_tasks_day ON completed_tasks(day);
CREATE INDEX IF NOT EXISTS idx_review_notes_user_id ON review_notes(user_id);
CREATE INDEX IF NOT EXISTS idx_review_notes_task_id ON review_notes(task_id);
CREATE INDEX IF NOT EXISTS idx_daily_progress_user_date ON daily_progress(user_id, date);

-- Function to auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers to auto-update updated_at
CREATE TRIGGER update_review_notes_updated_at BEFORE UPDATE ON review_notes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_daily_progress_updated_at BEFORE UPDATE ON daily_progress
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at BEFORE UPDATE ON user_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Enable Row Level Security (RLS)
ALTER TABLE completed_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE review_notes ENABLE ROW LEVEL SECURITY;
ALTER TABLE daily_progress ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;

-- Create policies (allow all operations for default_user)
CREATE POLICY "Allow all operations on completed_tasks" ON completed_tasks FOR ALL USING (true);
CREATE POLICY "Allow all operations on review_notes" ON review_notes FOR ALL USING (true);
CREATE POLICY "Allow all operations on daily_progress" ON daily_progress FOR ALL USING (true);
CREATE POLICY "Allow all operations on user_preferences" ON user_preferences FOR ALL USING (true);

-- Insert default preferences
INSERT INTO user_preferences (user_id, start_date, current_day, settings)
VALUES ('default_user', CURRENT_DATE, 1, '{"theme": "default", "notifications": true}')
ON CONFLICT (user_id) DO NOTHING;

SELECT 'Database schema created successfully! ✅' as message;

