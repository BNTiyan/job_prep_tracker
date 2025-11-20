// Application State
let state = {
    currentDay: 1,
    completedTasks: new Set(),
    customTopics: [],
    reviewNotes: [],
    startDate: getTodayDate(), // Start from today
    practiceProblems: null, // Will load from YAML
    learningModel: {
        performanceHistory: [], // Track daily performance
        categoryStrengths: {}, // Track strength in each category
        categoryWeaknesses: {}, // Track areas needing improvement
        preferences: {
            averageStudyTime: 120, // minutes
            preferredCategories: [],
            difficultTopics: []
        },
        adaptations: {
            suggestedFocus: [],
            recommendedReview: [],
            paceAdjustment: 'normal' // 'slower', 'normal', 'faster'
        }
    }
};

// Get today's date as start date
function getTodayDate() {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    return today.toISOString().split('T')[0];
}

// Get current day based on start date
function calculateCurrentDay() {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const start = new Date(state.startDate);
    start.setHours(0, 0, 0, 0);
    
    const diffTime = today - start;
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    
    // If today is before start date, we're at day 1 (preview mode)
    if (diffDays < 0) {
        return 1;
    }
    
    // Day 1 is the start date, so add 1
    return Math.min(diffDays + 1, 60);
}

// Check if we should auto-advance to today's day
function autoAdvanceToToday() {
    const calculatedDay = calculateCurrentDay();
    
    // Only auto-advance if user hasn't manually navigated
    if (!state.manualNavigation && calculatedDay > state.currentDay) {
        state.currentDay = calculatedDay;
        saveState();
    }
}

// Load practice problems from YAML
async function loadPracticeProblems() {
    try {
        const response = await fetch('practice_problems.yaml');
        const yamlText = await response.text();
        state.practiceProblems = jsyaml.load(yamlText);
        console.log('Practice problems loaded successfully');
    } catch (error) {
        console.error('Failed to load practice problems:', error);
        // Fallback to data.js if YAML loading fails
        if (typeof codingPractice !== 'undefined') {
            state.practiceProblems = {
                coding_practice: codingPractice,
                system_design_problems: systemDesignProblems,
                ml_coding_challenges: mlCodingChallenges
            };
        }
    }
}

// Load state from localStorage
function loadState() {
    const saved = localStorage.getItem('googlePrepState');
    if (saved) {
        const parsed = JSON.parse(saved);
        state = {
            ...state, // Keep default values
            ...parsed,
            completedTasks: new Set(parsed.completedTasks || []),
            reviewNotes: parsed.reviewNotes || []
        };
        
        // If no start date set, use today
        if (!state.startDate) {
            state.startDate = getTodayDate();
        }
    }
    
    // Auto-advance to today's day if needed
    autoAdvanceToToday();
}

// Save state to localStorage
function saveState() {
    localStorage.setItem('googlePrepState', JSON.stringify({
        ...state,
        completedTasks: Array.from(state.completedTasks)
    }));
    
    // Update learning model after each save
    updateLearningModel();
}

// Update learning model based on daily activities
function updateLearningModel() {
    const today = new Date().toISOString().split('T')[0];
    const dayPlan = preparationPlan.find(p => p.day === state.currentDay);
    
    if (!dayPlan) return;
    
    // Calculate today's performance
    const todayTasks = dayPlan.tasks.map((_, index) => `day-${state.currentDay}-task-${index}`);
    const completedToday = todayTasks.filter(taskId => state.completedTasks.has(taskId)).length;
    const completionRate = todayTasks.length > 0 ? (completedToday / todayTasks.length) * 100 : 0;
    
    // Calculate time spent (estimate based on completed tasks)
    const timeSpent = dayPlan.tasks
        .filter((_, index) => state.completedTasks.has(`day-${state.currentDay}-task-${index}`))
        .reduce((sum, task) => sum + task.duration, 0);
    
    // Record daily performance
    const dailyPerformance = {
        day: state.currentDay,
        date: today,
        category: dayPlan.category,
        completionRate: completionRate,
        tasksCompleted: completedToday,
        totalTasks: todayTasks.length,
        timeSpent: timeSpent,
        timestamp: Date.now()
    };
    
    // Update performance history (keep last 60 days)
    state.learningModel.performanceHistory = [
        ...state.learningModel.performanceHistory.filter(p => p.day !== state.currentDay),
        dailyPerformance
    ].slice(-60);
    
    // Update category strengths and weaknesses
    updateCategoryInsights(dayPlan.category, completionRate);
    
    // Update preferences
    updatePreferences(timeSpent, dayPlan.category);
    
    // Generate adaptive recommendations
    generateAdaptations();
}

// Update category-specific insights
function updateCategoryInsights(category, completionRate) {
    if (!state.learningModel.categoryStrengths[category]) {
        state.learningModel.categoryStrengths[category] = [];
    }
    
    state.learningModel.categoryStrengths[category].push(completionRate);
    
    // Keep only last 5 entries per category
    if (state.learningModel.categoryStrengths[category].length > 5) {
        state.learningModel.categoryStrengths[category].shift();
    }
    
    // Calculate average performance for this category
    const avgPerformance = state.learningModel.categoryStrengths[category].reduce((a, b) => a + b, 0) / 
                           state.learningModel.categoryStrengths[category].length;
    
    // Identify weaknesses (< 70% completion rate)
    if (avgPerformance < 70) {
        if (!state.learningModel.categoryWeaknesses[category]) {
            state.learningModel.categoryWeaknesses[category] = avgPerformance;
        }
    } else {
        delete state.learningModel.categoryWeaknesses[category];
    }
}

// Update user preferences based on behavior
function updatePreferences(timeSpent, category) {
    // Update average study time
    const recentTime = state.learningModel.performanceHistory.slice(-7)
        .reduce((sum, p) => sum + p.timeSpent, 0);
    const recentDays = state.learningModel.performanceHistory.slice(-7).length;
    
    if (recentDays > 0) {
        state.learningModel.preferences.averageStudyTime = Math.round(recentTime / recentDays);
    }
    
    // Track preferred categories (high completion rates)
    const categoryPerf = state.learningModel.categoryStrengths[category] || [];
    const avgCategoryPerf = categoryPerf.reduce((a, b) => a + b, 0) / (categoryPerf.length || 1);
    
    if (avgCategoryPerf >= 80) {
        if (!state.learningModel.preferences.preferredCategories.includes(category)) {
            state.learningModel.preferences.preferredCategories.push(category);
        }
    }
    
    // Track difficult topics
    if (avgCategoryPerf < 60) {
        if (!state.learningModel.preferences.difficultTopics.includes(category)) {
            state.learningModel.preferences.difficultTopics.push(category);
        }
    } else {
        state.learningModel.preferences.difficultTopics = 
            state.learningModel.preferences.difficultTopics.filter(t => t !== category);
    }
}

// Generate adaptive recommendations
function generateAdaptations() {
    const adaptations = {
        suggestedFocus: [],
        recommendedReview: [],
        paceAdjustment: 'normal'
    };
    
    // Analyze recent performance (last 7 days)
    const recentPerformance = state.learningModel.performanceHistory.slice(-7);
    const avgCompletionRate = recentPerformance.reduce((sum, p) => sum + p.completionRate, 0) / 
                               (recentPerformance.length || 1);
    
    // Pace adjustment
    if (avgCompletionRate >= 90) {
        adaptations.paceAdjustment = 'faster';
        adaptations.suggestedFocus.push('You\'re doing great! Consider adding advanced topics or harder problems.');
    } else if (avgCompletionRate < 60) {
        adaptations.paceAdjustment = 'slower';
        adaptations.suggestedFocus.push('Consider spending more time on fundamentals before moving forward.');
    }
    
    // Identify topics needing review
    const weakCategories = Object.keys(state.learningModel.categoryWeaknesses)
        .sort((a, b) => state.learningModel.categoryWeaknesses[a] - state.learningModel.categoryWeaknesses[b])
        .slice(0, 3);
    
    if (weakCategories.length > 0) {
        adaptations.recommendedReview = weakCategories;
        adaptations.suggestedFocus.push(
            `Focus on: ${weakCategories.join(', ')}. Your completion rate in these areas is below 70%.`
        );
    }
    
    // Suggest balance
    if (state.learningModel.preferences.difficultTopics.length > 3) {
        adaptations.suggestedFocus.push(
            'Mix difficult topics with areas you\'re strong in to maintain motivation.'
        );
    }
    
    // Time management insights
    const avgTime = state.learningModel.preferences.averageStudyTime;
    if (avgTime < 90) {
        adaptations.suggestedFocus.push(
            `You're averaging ${avgTime} min/day. Try to reach the 2-hour (120 min) daily goal for optimal prep.`
        );
    } else if (avgTime >= 120) {
        adaptations.suggestedFocus.push(
            'Great consistency! You\'re meeting your 2-hour daily target. Keep it up! 🎉'
        );
    }
    
    // Suggest coding practice based on gaps
    const codingDays = recentPerformance.filter(p => 
        p.category.includes('Coding') || p.category.includes('Algorithms')
    ).length;
    
    if (codingDays === 0 && recentPerformance.length >= 5) {
        adaptations.suggestedFocus.push(
            'You haven\'t practiced coding in a while. Try solving 2-3 LeetCode problems today.'
        );
    }
    
    state.learningModel.adaptations = adaptations;
}

// Initialize app
async function init() {
    await loadPracticeProblems();
    loadState();
    
    // Show welcome message for new users
    showWelcomeIfNeeded();
    
    renderTasks();
    updateProgress();
    updateDateDisplay();
    renderInsights(); // Show AI-powered insights
    setupEventListeners();
}

// Show welcome message for first-time users
function showWelcomeIfNeeded() {
    const hasSeenWelcome = localStorage.getItem('hasSeenWelcome');
    
    if (!hasSeenWelcome) {
        const startDate = new Date(state.startDate);
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        
        const daysUntilStart = Math.max(0, Math.ceil((startDate - today) / (1000 * 60 * 60 * 24)));
        const startMessage = daysUntilStart > 0
            ? `starts in <strong>${daysUntilStart} day${daysUntilStart > 1 ? 's' : ''}</strong>`
            : 'starts <strong>today</strong>';
        const noteMessage = daysUntilStart > 0
            ? 'Countdown is on—preview Day 1 now and get ready to hit the ground running.'
            : 'We’re kicking off right now. Open today’s plan and start building momentum. 🚀';
        
        const welcomeMessage = `
            <div class="welcome-overlay" id="welcomeOverlay">
                <div class="welcome-modal">
                    <h2>🎯 Welcome to Your AI/ML Interview Prep!</h2>
                    <p>Your 60-day preparation journey ${startMessage}!</p>
                    <div class="welcome-info">
                        <div class="welcome-item">
                            <span class="welcome-icon">📅</span>
                            <div>
                                <strong>Start Date:</strong> ${startDate.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })}
                            </div>
                        </div>
                        <div class="welcome-item">
                            <span class="welcome-icon">⏰</span>
                            <div>
                                <strong>Daily Commitment:</strong> 2 hours of focused study
                            </div>
                        </div>
                        <div class="welcome-item">
                            <span class="welcome-icon">🤖</span>
                            <div>
                                <strong>AI Learning:</strong> Personalized insights as you progress
                            </div>
                        </div>
                        <div class="welcome-item">
                            <span class="welcome-icon">💻</span>
                            <div>
                                <strong>What's Included:</strong> Coding problems, ML theory, System design, Implementation challenges
                            </div>
                        </div>
                    </div>
                    <p class="welcome-note">${noteMessage}</p>
                    <button id="welcomeStartBtn" class="btn btn-primary btn-large">
                        Let's Get Started! 💪
                    </button>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', welcomeMessage);
        
        document.getElementById('welcomeStartBtn').addEventListener('click', () => {
            localStorage.setItem('hasSeenWelcome', 'true');
            document.getElementById('welcomeOverlay').remove();
        });
    }
}

// Render AI-powered insights
function renderInsights() {
    const container = document.getElementById('insightsContainer');
    if (!container) return;
    
    container.innerHTML = '';
    
    const adaptations = state.learningModel.adaptations;
    const preferences = state.learningModel.preferences;
    const recentPerformance = state.learningModel.performanceHistory.slice(-7);
    
    if (recentPerformance.length === 0) {
        container.innerHTML = `
            <div class="insight-card">
                <h3>🚀 Welcome to Your AI-Powered Study Plan!</h3>
                <p>As you complete tasks, I'll learn your strengths and adapt recommendations.</p>
                <p>Start completing today's tasks to see personalized insights!</p>
            </div>
        `;
        return;
    }
    
    // Performance summary
    const avgCompletion = recentPerformance.reduce((sum, p) => sum + p.completionRate, 0) / recentPerformance.length;
    const totalTime = recentPerformance.reduce((sum, p) => sum + p.timeSpent, 0);
    
    let performanceEmoji = '📈';
    if (avgCompletion >= 90) performanceEmoji = '🔥';
    else if (avgCompletion >= 70) performanceEmoji = '💪';
    else if (avgCompletion < 50) performanceEmoji = '⚠️';
    
    container.innerHTML = `
        <div class="insight-card">
            <h3>${performanceEmoji} Your Learning Progress (Last 7 Days)</h3>
            <div class="insight-stats">
                <div class="insight-stat">
                    <span class="insight-value">${Math.round(avgCompletion)}%</span>
                    <span class="insight-label">Avg Completion</span>
                </div>
                <div class="insight-stat">
                    <span class="insight-value">${Math.round(totalTime / recentPerformance.length)}m</span>
                    <span class="insight-label">Avg Daily Time</span>
                </div>
                <div class="insight-stat">
                    <span class="insight-value">${recentPerformance.length}</span>
                    <span class="insight-label">Active Days</span>
                </div>
            </div>
        </div>
        
        ${adaptations.suggestedFocus.length > 0 ? `
        <div class="insight-card">
            <h3>💡 AI Recommendations</h3>
            <ul class="insight-list">
                ${adaptations.suggestedFocus.map(suggestion => `<li>${suggestion}</li>`).join('')}
            </ul>
        </div>
        ` : ''}
        
        ${adaptations.recommendedReview.length > 0 ? `
        <div class="insight-card insight-warning">
            <h3>📚 Topics to Review</h3>
            <p>Based on your performance, focus on:</p>
            <ul class="insight-list">
                ${adaptations.recommendedReview.map(topic => `<li><strong>${topic}</strong></li>`).join('')}
            </ul>
        </div>
        ` : ''}
        
        ${preferences.preferredCategories.length > 0 ? `
        <div class="insight-card insight-success">
            <h3>⭐ Your Strengths</h3>
            <p>You're excelling in: ${preferences.preferredCategories.join(', ')}</p>
        </div>
        ` : ''}
        
        <div class="insight-card">
            <h3>📊 Pace: ${adaptations.paceAdjustment === 'faster' ? '🚀 Ahead' : 
                            adaptations.paceAdjustment === 'slower' ? '🐢 Take Your Time' : 
                            '✅ On Track'}</h3>
            <p>${adaptations.paceAdjustment === 'faster' ? 
                'You\'re ahead of schedule! Consider tackling more challenging problems.' :
                adaptations.paceAdjustment === 'slower' ?
                'No rush! Focus on understanding concepts deeply before moving forward.' :
                'You\'re progressing at a steady pace. Keep up the good work!'}</p>
        </div>
    `;
}

// Render tasks for current day
function renderTasks() {
    const container = document.getElementById('tasksContainer');
    container.innerHTML = '';

    // Get tasks for current day
    const dayPlan = preparationPlan.find(p => p.day === state.currentDay);
    
    if (!dayPlan) {
        container.innerHTML = '<div class="task-category"><p>No tasks planned for this day yet. Add custom topics!</p></div>';
        renderNotesSection();
        return;
    }

    // Get uncompleted tasks from previous days
    const uncompletedPrevious = getUncompletedPreviousTasks();

    // Render current day tasks
    const categoryCard = createCategoryCard(dayPlan, state.currentDay);
    container.appendChild(categoryCard);

    // Render custom topics for this day
    const customForToday = state.customTopics.filter(t => t.day === state.currentDay);
    if (customForToday.length > 0) {
        const customCard = createCustomTopicsCard(customForToday, state.currentDay);
        container.appendChild(customCard);
    }

    // Render uncompleted previous tasks
    if (uncompletedPrevious.length > 0) {
        const uncompletedCard = createUncompletedCard(uncompletedPrevious);
        container.appendChild(uncompletedCard);
    }

    renderNotesSection();
}

// Get uncompleted tasks from previous days
function getUncompletedPreviousTasks() {
    const uncompleted = [];
    
    for (let day = 1; day < state.currentDay; day++) {
        const dayPlan = preparationPlan.find(p => p.day === day);
        if (dayPlan) {
            dayPlan.tasks.forEach((task, index) => {
                const taskId = `day-${day}-task-${index}`;
                if (!state.completedTasks.has(taskId)) {
                    uncompleted.push({ ...task, originalDay: day, taskId });
                }
            });
        }

        // Check custom topics
        const customForDay = state.customTopics.filter(t => t.day === day);
        customForDay.forEach(topic => {
            if (!state.completedTasks.has(topic.id)) {
                uncompleted.push({ ...topic, originalDay: day, taskId: topic.id });
            }
        });
    }

    return uncompleted;
}

// Create category card
function createCategoryCard(dayPlan, day) {
    const card = document.createElement('div');
    card.className = 'task-category';

    const totalDuration = dayPlan.tasks.reduce((sum, task) => sum + task.duration, 0);
    
    card.innerHTML = `
        <div class="category-header">
            <h3 class="category-title">
                ${getCategoryIcon(dayPlan.category)} ${dayPlan.category}
            </h3>
            <span class="category-badge">${totalDuration} minutes</span>
        </div>
        <div class="task-list" id="taskList-${day}"></div>
    `;

    const taskList = card.querySelector(`#taskList-${day}`);
    dayPlan.tasks.forEach((task, index) => {
        const taskId = `day-${day}-task-${index}`;
        const taskElement = createTaskElement(task, taskId);
        taskList.appendChild(taskElement);
    });

    return card;
}

// Create custom topics card
function createCustomTopicsCard(topics, day) {
    const card = document.createElement('div');
    card.className = 'task-category';
    
    const totalDuration = topics.reduce((sum, t) => sum + t.duration, 0);
    
    card.innerHTML = `
        <div class="category-header">
            <h3 class="category-title">
                ⭐ Custom Topics
            </h3>
            <span class="category-badge">${totalDuration} minutes</span>
        </div>
        <div class="task-list" id="customTaskList-${day}"></div>
    `;

    const taskList = card.querySelector(`#customTaskList-${day}`);
    topics.forEach(topic => {
        const taskElement = createTaskElement({
            title: topic.name,
            duration: topic.duration,
            notes: topic.notes
        }, topic.id);
        taskList.appendChild(taskElement);
    });

    return card;
}

// Create uncompleted tasks card
function createUncompletedCard(tasks) {
    const card = document.createElement('div');
    card.className = 'task-category';
    card.style.borderLeft = '4px solid var(--warning-color)';
    
    card.innerHTML = `
        <div class="category-header">
            <h3 class="category-title">
                ⚠️ Carried Over from Previous Days
            </h3>
            <span class="category-badge">${tasks.length} tasks</span>
        </div>
        <div class="task-list" id="uncompletedTaskList"></div>
    `;

    const taskList = card.querySelector('#uncompletedTaskList');
    tasks.forEach(task => {
        const taskElement = createTaskElement({
            title: `[Day ${task.originalDay}] ${task.title}`,
            duration: task.duration,
            notes: task.notes
        }, task.taskId);
        taskList.appendChild(taskElement);
    });

    return card;
}

// Create individual task element
function createTaskElement(task, taskId) {
    const taskDiv = document.createElement('div');
    taskDiv.className = 'task-item';
    
    const isCompleted = state.completedTasks.has(taskId);
    if (isCompleted) {
        taskDiv.classList.add('completed');
    }

    taskDiv.innerHTML = `
        <input type="checkbox" class="task-checkbox" ${isCompleted ? 'checked' : ''} data-task-id="${taskId}">
        <div class="task-content">
            <div class="task-title">${task.title}</div>
            <span class="task-duration">⏱️ ${task.duration} minutes</span>
            ${task.notes ? `<div class="task-notes">💡 ${task.notes}</div>` : ''}
        </div>
    `;

    return taskDiv;
}

// Render notes / review section
function renderNotesSection() {
    const section = document.getElementById('notesSection');
    if (!section) return;

    if (!Array.isArray(state.reviewNotes)) {
        state.reviewNotes = [];
    }

    const notes = [...state.reviewNotes].sort((a, b) => new Date(b.createdAt || 0) - new Date(a.createdAt || 0));
    const dayOptions = Array.from({ length: 60 }, (_, i) => {
        const dayNumber = i + 1;
        const selected = dayNumber === state.currentDay ? 'selected' : '';
        return `<option value="${dayNumber}" ${selected}>Day ${dayNumber}</option>`;
    }).join('');

    const notesList = notes.length
        ? notes.map(note => `
            <div class="note-card">
                <div class="note-card-header">
                    <span class="note-day">Day ${note.day}</span>
                    <span class="note-date">${formatNoteDate(note.createdAt)}</span>
                    <button class="note-delete" data-note-id="${note.id}" aria-label="Delete note">✕</button>
                </div>
                <p>${escapeHtml(note.text)}</p>
            </div>
        `).join('')
        : `<p class="notes-empty">No review notes yet. Capture tricky tasks to revisit later.</p>`;

    section.innerHTML = `
        <div class="notes-header">
            <h2>🔖 Review Notes</h2>
            <p>Keep track of follow-ups, tricky tasks, or resources to review.</p>
        </div>
        <form id="noteForm" class="note-form">
            <div class="note-form-row">
                <div class="form-group note-day-select">
                    <label for="noteDay">Day</label>
                    <select id="noteDay">${dayOptions}</select>
                </div>
                <div class="form-group flex-grow">
                    <label for="noteText">Note</label>
                    <textarea id="noteText" rows="2" placeholder="Example: Revisit DP derivation for LIS" required></textarea>
                </div>
            </div>
            <button type="submit" class="btn btn-primary">Save Note</button>
        </form>
        <div class="notes-list">
            ${notesList}
        </div>
    `;

    const noteForm = document.getElementById('noteForm');
    if (noteForm) {
        noteForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const noteTextEl = document.getElementById('noteText');
            const noteDayEl = document.getElementById('noteDay');
            const noteText = noteTextEl.value.trim();
            if (!noteText) return;

            const note = {
                id: 'note-' + Date.now(),
                day: Math.min(Math.max(parseInt(noteDayEl.value, 10) || state.currentDay, 1), 60),
                text: noteText,
                createdAt: new Date().toISOString()
            };

            state.reviewNotes.push(note);
            noteTextEl.value = '';
            saveState();
            renderNotesSection();
        });
    }

    section.querySelectorAll('.note-delete').forEach(button => {
        button.addEventListener('click', () => {
            const noteId = button.dataset.noteId;
            state.reviewNotes = state.reviewNotes.filter(note => note.id !== noteId);
            saveState();
            renderNotesSection();
        });
    });
}

function formatNoteDate(dateString) {
    if (!dateString) return '';
    const date = new Date(dateString);
    if (Number.isNaN(date.getTime())) return '';
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function escapeHtml(text = '') {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// Get category icon
function getCategoryIcon(category) {
    const icons = {
        'Machine Learning Fundamentals': '🤖',
        'Deep Learning Fundamentals': '🧠',
        'Deep Learning - CNNs': '👁️',
        'Computer Vision Advanced': '📷',
        'Deep Learning - RNNs & LSTMs': '🔄',
        'Natural Language Processing': '💬',
        'NLP & LLMs': '🗣️',
        'MLOps & Production ML': '⚙️',
        'Behavioral - ML Projects': '🎯',
        'ML System Design': '🏗️',
        'ML Algorithms - Unsupervised': '🔍',
        'ML Theory & Math': '📐',
        'Reinforcement Learning': '🎮',
        'ML Coding Interview': '💻',
        'Mock Interview - ML': '🎭',
        'Graph Neural Networks': '🕸️',
        'Generative AI': '🎨',
        'Model Optimization & Deployment': '🚀',
        'ML Security & Safety': '🔒',
        'Time Series & Forecasting': '📈',
        'Data Engineering for ML': '🗄️',
        'Distributed ML': '☁️',
        'ML Model Monitoring': '📊',
        'ML System Design - Advanced': '🏛️',
        'Mock Interview - Advanced ML': '🎪',
        'Google ML Infrastructure': '🔧',
        'Google ML System Design': '🌐',
        'ML Algorithms - Advanced': '🧮',
        'Causal Inference & Experimentation': '🔬',
        'Behavioral - Googliness': '✨',
        'ML Coding Interview Prep': '⌨️',
        'ML Theory Interview': '📚',
        'ML System Design - Mock': '🎓',
        'Data Structures & Algorithms for ML Engineers': '💡',
        'Full Mock Interview Day': '🏆',
        'ML Projects Deep Dive': '🔨',
        'ML Case Studies': '📋',
        'ML System Design - Real Cases': '🌟',
        'Mock Interview - Intensive': '🔥',
        'Advanced NLP': '📝',
        'Advanced Computer Vision': '🖼️',
        'ML at Scale': '📡',
        'ML Infrastructure & Tools': '🛠️',
        'Mock Interview - Final Round': '🎬',
        'Resume & Portfolio Polish': '✍️',
        'ML Research Awareness': '🔭',
        'Behavioral Interview Polish': '💼',
        'ML Coding Speed Practice': '⚡',
        'ML System Design Speed Practice': '🏃',
        'Light Review - ML Theory': '📖',
        'Light Review - Your Experience': '👤',
        'Mental Preparation': '🧘',
        'Final Day Before Interview': '📅',
        'Interview Day': '🎉'
    };
    return icons[category] || '📚';
}

// Update progress statistics
function updateProgress() {
    const totalTasks = getTotalTasks();
    const completed = state.completedTasks.size;
    const percentage = totalTasks > 0 ? Math.round((completed / totalTasks) * 100) : 0;

    document.getElementById('progressFill').style.width = percentage + '%';
    document.getElementById('progressText').textContent = percentage + '%';
    document.getElementById('completedTasks').textContent = completed;
    document.getElementById('totalTasks').textContent = totalTasks;
    
    // Calculate streak
    const streak = calculateStreak();
    document.getElementById('currentStreak').textContent = streak;
}

// Get total number of tasks
function getTotalTasks() {
    let total = 0;
    
    // Count predefined tasks up to current day
    for (let day = 1; day <= state.currentDay; day++) {
        const dayPlan = preparationPlan.find(p => p.day === day);
        if (dayPlan) {
            total += dayPlan.tasks.length;
        }
    }
    
    // Add custom topics up to current day
    total += state.customTopics.filter(t => t.day <= state.currentDay).length;
    
    return total;
}

// Calculate current streak
function calculateStreak() {
    let streak = 0;
    const today = new Date();
    
    for (let i = 0; i < 60; i++) {
        const checkDate = new Date(today);
        checkDate.setDate(today.getDate() - i);
        const dayTasks = getTasksForDate(checkDate);
        
        if (dayTasks.length > 0 && dayTasks.every(taskId => state.completedTasks.has(taskId))) {
            streak++;
        } else {
            break;
        }
    }
    
    return streak;
}

// Get tasks for a specific date
function getTasksForDate(date) {
    const daysDiff = Math.floor((date - new Date(state.startDate)) / (1000 * 60 * 60 * 24)) + 1;
    const tasks = [];
    
    const dayPlan = preparationPlan.find(p => p.day === daysDiff);
    if (dayPlan) {
        dayPlan.tasks.forEach((_, index) => {
            tasks.push(`day-${daysDiff}-task-${index}`);
        });
    }
    
    state.customTopics.filter(t => t.day === daysDiff).forEach(topic => {
        tasks.push(topic.id);
    });
    
    return tasks;
}

// Update date display
function updateDateDisplay() {
    const date = new Date(state.startDate);
    date.setDate(date.getDate() + state.currentDay - 1);
    
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const currentDate = new Date(date);
    currentDate.setHours(0, 0, 0, 0);
    
    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    const dateString = currentDate.toLocaleDateString('en-US', options);
    
    // Calculate days until start or days into plan
    const startDate = new Date(state.startDate);
    startDate.setHours(0, 0, 0, 0);
    const daysUntilStart = Math.max(0, Math.ceil((startDate - today) / (1000 * 60 * 60 * 24)));
    
    let dateLabel = '';
    if (daysUntilStart > 0 && state.currentDay === 1) {
        dateLabel = `<span class="date-badge date-future">Starts in ${daysUntilStart} day${daysUntilStart > 1 ? 's' : ''}</span>`;
    } else if (currentDate.getTime() === today.getTime()) {
        dateLabel = '<span class="date-badge date-today">Today 📍</span>';
    } else if (currentDate < today) {
        dateLabel = '<span class="date-badge date-past">Past Day</span>';
    } else {
        dateLabel = '<span class="date-badge date-future">Upcoming</span>';
    }
    
    document.getElementById('currentDate').innerHTML = `
        Day ${state.currentDay} - ${dateString}
        ${dateLabel}
    `;
    
    // Update navigation buttons
    document.getElementById('prevDay').disabled = state.currentDay <= 1;
    document.getElementById('nextDay').disabled = state.currentDay >= 60;
}

// Setup event listeners
function setupEventListeners() {
    // Navigation buttons
    document.getElementById('prevDay').addEventListener('click', () => {
        if (state.currentDay > 1) {
            state.currentDay--;
            state.manualNavigation = true; // User manually navigated
            saveState();
            renderTasks();
            updateProgress();
            updateDateDisplay();
            renderInsights();
        }
    });

    document.getElementById('nextDay').addEventListener('click', () => {
        if (state.currentDay < 60) {
            state.currentDay++;
            state.manualNavigation = true; // User manually navigated
            saveState();
            renderTasks();
            updateProgress();
            updateDateDisplay();
            renderInsights();
        }
    });

    // Jump to today button
    document.getElementById('todayBtn').addEventListener('click', () => {
        const todayDay = calculateCurrentDay();
        if (todayDay !== state.currentDay) {
            state.currentDay = todayDay;
            state.manualNavigation = false; // Reset manual navigation flag
            saveState();
            renderTasks();
            updateProgress();
            updateDateDisplay();
            renderInsights();
        }
    });

    // Task checkboxes
    document.addEventListener('change', (e) => {
        if (e.target.classList.contains('task-checkbox')) {
            const taskId = e.target.dataset.taskId;
            if (e.target.checked) {
                state.completedTasks.add(taskId);
            } else {
                state.completedTasks.delete(taskId);
            }
            saveState();
            renderTasks();
            updateProgress();
            renderInsights(); // Update AI insights when tasks change
        }
    });

    // Add custom topic modal
    const customModal = document.getElementById('customTopicModal');
    const addBtn = document.getElementById('addTopicBtn');
    const closeCustom = document.querySelector('#customTopicModal .close');

    addBtn.onclick = () => {
        customModal.style.display = 'block';
    };

    closeCustom.onclick = () => {
        customModal.style.display = 'none';
    };

    // Learning topics modal
    const topicsModal = document.getElementById('learningTopicsModal');
    const viewTopicsBtn = document.getElementById('viewTopicsBtn');
    const closeTopics = document.querySelector('.close-topics');

    viewTopicsBtn.onclick = () => {
        renderLearningTopics();
        topicsModal.style.display = 'block';
    };

    closeTopics.onclick = () => {
        topicsModal.style.display = 'none';
    };

    // Coding problems modal
    const problemsModal = document.getElementById('codingProblemsModal');
    const viewProblemsBtn = document.getElementById('viewProblemsBtn');
    const closeProblems = document.querySelector('.close-problems');

    viewProblemsBtn.onclick = () => {
        renderCodingProblems();
        problemsModal.style.display = 'block';
    };

    closeProblems.onclick = () => {
        problemsModal.style.display = 'none';
    };

    // System design modal
    const designModal = document.getElementById('systemDesignModal');
    const viewDesignBtn = document.getElementById('viewSystemDesignBtn');
    const closeDesign = document.querySelector('.close-design');

    viewDesignBtn.onclick = () => {
        renderSystemDesignProblems();
        designModal.style.display = 'block';
    };

    closeDesign.onclick = () => {
        designModal.style.display = 'none';
    };

    // Analytics modal
    const analyticsModal = document.getElementById('analyticsModal');
    const viewAnalyticsBtn = document.getElementById('viewAnalyticsBtn');
    const closeAnalytics = document.querySelector('.close-analytics');

    viewAnalyticsBtn.onclick = () => {
        renderAnalytics();
        analyticsModal.style.display = 'block';
    };

    closeAnalytics.onclick = () => {
        analyticsModal.style.display = 'none';
    };

    // Export data button
    document.getElementById('exportDataBtn').addEventListener('click', exportLearningData);

    window.onclick = (event) => {
        if (event.target == customModal) {
            customModal.style.display = 'none';
        }
        if (event.target == topicsModal) {
            topicsModal.style.display = 'none';
        }
        if (event.target == problemsModal) {
            problemsModal.style.display = 'none';
        }
        if (event.target == designModal) {
            designModal.style.display = 'none';
        }
        if (event.target == analyticsModal) {
            analyticsModal.style.display = 'none';
        }
    };

    // Custom topic form submission
    document.getElementById('customTopicForm').addEventListener('submit', (e) => {
        e.preventDefault();
        
        const topic = {
            id: 'custom-' + Date.now(),
            day: state.currentDay,
            name: document.getElementById('topicName').value,
            duration: parseInt(document.getElementById('topicDuration').value),
            category: document.getElementById('topicCategory').value,
            notes: document.getElementById('topicNotes').value
        };

        state.customTopics.push(topic);
        saveState();
        
        customModal.style.display = 'none';
        document.getElementById('customTopicForm').reset();
        
        renderTasks();
        updateProgress();
    });

    // Reset progress button
    document.getElementById('resetProgressBtn').addEventListener('click', () => {
        if (confirm('Are you sure you want to reset all progress? This cannot be undone.')) {
            state = {
                currentDay: 1,
                completedTasks: new Set(),
                customTopics: [],
                reviewNotes: [],
                startDate: getTodayDate()
            };
            saveState();
            renderTasks();
            updateProgress();
            updateDateDisplay();
            renderNotesSection();
        }
    });
}

// Render learning topics reference
function renderLearningTopics() {
    const container = document.getElementById('learningTopicsContainer');
    container.innerHTML = '';

    // Check if learningTopics is defined (from data.js)
    if (typeof learningTopics === 'undefined') {
        container.innerHTML = '<p>Learning topics not available.</p>';
        return;
    }

    for (const [category, topics] of Object.entries(learningTopics)) {
        const categoryCard = document.createElement('div');
        categoryCard.className = 'topic-category-card';
        
        const icon = getCategoryIconForLearningTopic(category);
        
        categoryCard.innerHTML = `
            <h3 class="topic-category-title">${icon} ${category}</h3>
            <ul class="topic-list">
                ${topics.map(topic => `<li>${topic}</li>`).join('')}
            </ul>
        `;
        
        container.appendChild(categoryCard);
    }
}

// Get category icon for learning topics
function getCategoryIconForLearningTopic(category) {
    const icons = {
        'Machine Learning Fundamentals': '🤖',
        'Deep Learning': '🧠',
        'Computer Vision': '📷',
        'Natural Language Processing': '💬',
        'Reinforcement Learning': '🎮',
        'MLOps & Production ML': '⚙️',
        'ML System Design': '🏗️',
        'Advanced Topics': '🚀',
        'ML Math & Theory': '📐',
        'Tools & Frameworks': '🛠️',
        'Data Engineering for ML': '🗄️',
        'ML Security & Ethics': '🔒'
    };
    return icons[category] || '📚';
}

// Render coding problems
function renderCodingProblems() {
    const container = document.getElementById('codingProblemsContainer');
    container.innerHTML = '';

    if (!state.practiceProblems || !state.practiceProblems.coding_practice) {
        container.innerHTML = '<p>Practice problems are loading...</p>';
        return;
    }

    const problems = state.practiceProblems.coding_practice;
    
    // Group by week
    const weeks = {};
    problems.forEach(dayProblems => {
        const week = Math.ceil(dayProblems.day / 7);
        if (!weeks[week]) weeks[week] = [];
        weeks[week].push(dayProblems);
    });

    for (const [week, dayProblems] of Object.entries(weeks).sort((a, b) => a[0] - b[0])) {
        const weekSection = document.createElement('div');
        weekSection.className = 'week-section';
        weekSection.innerHTML = `<h3 style="margin: 20px 0 15px 0; color: var(--primary-color);">Week ${week}</h3>`;
        
        dayProblems.forEach(day => {
            const card = document.createElement('div');
            card.className = 'problem-card';
            
            const difficultyClass = `difficulty-${day.difficulty.toLowerCase().split(' ')[0]}`;
            
            let problemsList = '';
            if (day.problems && Array.isArray(day.problems)) {
                problemsList = '<ul class="problem-list">';
                day.problems.forEach(problem => {
                    const link = problem.link || '#';
                    const topics = problem.topics ? ` (${problem.topics.join(', ')})` : '';
                    problemsList += `<li><a href="${link}" target="_blank" class="problem-link">${problem.name}</a> - ${problem.difficulty}${topics}</li>`;
                });
                problemsList += '</ul>';
            }
            
            card.innerHTML = `
                <div class="problem-header">
                    <div class="problem-title">Day ${day.day}: ${day.category}</div>
                    <span class="problem-difficulty ${difficultyClass}">${day.difficulty}</span>
                </div>
                ${problemsList}
            `;
            
            weekSection.appendChild(card);
        });
        
        container.appendChild(weekSection);
    }
}

// Render system design problems
function renderSystemDesignProblems() {
    const container = document.getElementById('systemDesignContainer');
    container.innerHTML = '';

    if (!state.practiceProblems || !state.practiceProblems.system_design_problems) {
        container.innerHTML = '<p>System design problems are loading...</p>';
        return;
    }

    const problems = state.practiceProblems.system_design_problems;

    problems.forEach(problem => {
        const card = document.createElement('div');
        card.className = 'problem-card';
        
        const difficultyClass = `difficulty-${problem.difficulty.toLowerCase().split('-')[0]}`;
        
        let focusAreas = '';
        if (problem.focus_areas && Array.isArray(problem.focus_areas)) {
            focusAreas = '<ul class="problem-list">';
            problem.focus_areas.forEach(area => {
                focusAreas += `<li>${area}</li>`;
            });
            focusAreas += '</ul>';
        }
        
        card.innerHTML = `
            <div class="problem-header">
                <div class="problem-title">Day ${problem.day}: ${problem.problem}</div>
                <span class="problem-difficulty ${difficultyClass}">${problem.difficulty}</span>
            </div>
            <div class="problem-info">
                <span class="problem-tag">⏱️ ${problem.estimate_time}</span>
                ${problem.related_experience ? `<span class="problem-tag">💡 ${problem.related_experience}</span>` : ''}
            </div>
            <div style="margin-top: 15px;">
                <strong>Key Focus Areas:</strong>
                ${focusAreas}
            </div>
        `;
        
        container.appendChild(card);
    });
}

// Render detailed analytics
function renderAnalytics() {
    const container = document.getElementById('analyticsContent');
    container.innerHTML = '';

    const performanceHistory = state.learningModel.performanceHistory;
    const categoryStrengths = state.learningModel.categoryStrengths;
    const preferences = state.learningModel.preferences;
    const adaptations = state.learningModel.adaptations;

    if (performanceHistory.length === 0) {
        container.innerHTML = `
            <div class="analytics-empty">
                <h3>📈 No Data Yet</h3>
                <p>Start completing tasks to see detailed analytics and AI-powered insights!</p>
            </div>
        `;
        return;
    }

    // Overall Statistics
    const totalDays = performanceHistory.length;
    const totalTasks = performanceHistory.reduce((sum, p) => sum + p.tasksCompleted, 0);
    const totalTime = performanceHistory.reduce((sum, p) => sum + p.timeSpent, 0);
    const avgCompletion = performanceHistory.reduce((sum, p) => sum + p.completionRate, 0) / totalDays;

    // Category breakdown
    let categoryHTML = '<h3>📊 Performance by Category</h3><div class="category-grid">';
    for (const [category, scores] of Object.entries(categoryStrengths)) {
        const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
        const barColor = avgScore >= 80 ? 'var(--success-color)' : avgScore >= 60 ? 'var(--warning-color)' : 'var(--danger-color)';
        
        categoryHTML += `
            <div class="category-stat">
                <div class="category-name">${category}</div>
                <div class="category-bar">
                    <div class="category-bar-fill" style="width: ${avgScore}%; background: ${barColor};"></div>
                </div>
                <div class="category-score">${Math.round(avgScore)}%</div>
            </div>
        `;
    }
    categoryHTML += '</div>';

    // Recent performance trend
    const recentTrend = performanceHistory.slice(-14);
    let trendHTML = '<h3>📈 Performance Trend (Last 14 Days)</h3><div class="trend-chart">';
    recentTrend.forEach(day => {
        const barHeight = day.completionRate;
        const barColor = barHeight >= 80 ? 'var(--success-color)' : barHeight >= 60 ? 'var(--warning-color)' : 'var(--danger-color)';
        trendHTML += `
            <div class="trend-bar-container">
                <div class="trend-bar" style="height: ${barHeight}%; background: ${barColor};" 
                     title="Day ${day.day}: ${Math.round(day.completionRate)}%"></div>
                <span class="trend-label">D${day.day}</span>
            </div>
        `;
    });
    trendHTML += '</div>';

    // Study time analysis
    const timeByDay = {};
    performanceHistory.forEach(p => {
        const dayOfWeek = new Date(p.date).getDay();
        if (!timeByDay[dayOfWeek]) timeByDay[dayOfWeek] = [];
        timeByDay[dayOfWeek].push(p.timeSpent);
    });

    const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    let timeHTML = '<h3>⏰ Study Time by Day of Week</h3><div class="time-chart">';
    for (let day = 0; day < 7; day++) {
        const avgTime = timeByDay[day] ? 
            timeByDay[day].reduce((a, b) => a + b, 0) / timeByDay[day].length : 0;
        timeHTML += `
            <div class="time-day">
                <div class="time-label">${dayNames[day]}</div>
                <div class="time-value">${Math.round(avgTime)}m</div>
            </div>
        `;
    }
    timeHTML += '</div>';

    container.innerHTML = `
        <div class="analytics-overview">
            <h3>🎯 Overall Statistics</h3>
            <div class="analytics-stats">
                <div class="analytics-stat">
                    <span class="analytics-value">${totalDays}</span>
                    <span class="analytics-label">Days Active</span>
                </div>
                <div class="analytics-stat">
                    <span class="analytics-value">${totalTasks}</span>
                    <span class="analytics-label">Tasks Completed</span>
                </div>
                <div class="analytics-stat">
                    <span class="analytics-value">${Math.round(totalTime / 60)}h</span>
                    <span class="analytics-label">Total Study Time</span>
                </div>
                <div class="analytics-stat">
                    <span class="analytics-value">${Math.round(avgCompletion)}%</span>
                    <span class="analytics-label">Avg Completion</span>
                </div>
            </div>
        </div>

        ${categoryHTML}
        ${trendHTML}
        ${timeHTML}

        <div class="analytics-insights">
            <h3>🤖 AI Learning Model Insights</h3>
            
            ${preferences.preferredCategories.length > 0 ? `
            <div class="insight-section">
                <h4>✅ Your Strengths:</h4>
                <p>${preferences.preferredCategories.join(', ')}</p>
            </div>
            ` : ''}

            ${preferences.difficultTopics.length > 0 ? `
            <div class="insight-section">
                <h4>⚠️ Areas Needing Focus:</h4>
                <p>${preferences.difficultTopics.join(', ')}</p>
            </div>
            ` : ''}

            <div class="insight-section">
                <h4>🎯 Current Pace:</h4>
                <p>${adaptations.paceAdjustment === 'faster' ? '🚀 You\'re ahead of schedule! Consider advanced topics.' :
                     adaptations.paceAdjustment === 'slower' ? '🐢 Take your time to master fundamentals.' :
                     '✅ You\'re progressing at a perfect pace!'}</p>
            </div>

            <div class="insight-section">
                <h4>💡 Personalized Recommendations:</h4>
                <ul>
                    ${adaptations.suggestedFocus.map(s => `<li>${s}</li>`).join('')}
                </ul>
            </div>

            <div class="insight-section">
                <h4>📅 Study Pattern:</h4>
                <p>Average daily study time: <strong>${preferences.averageStudyTime} minutes</strong></p>
                <p>${preferences.averageStudyTime >= 120 ? 
                    '🎉 Excellent! You\'re meeting the 2-hour daily goal.' :
                    `📈 Try to reach 120 minutes daily for optimal preparation.`}</p>
            </div>
        </div>
    `;
}

// Export learning data
function exportLearningData() {
    const exportData = {
        exportDate: new Date().toISOString(),
        summary: {
            totalDays: state.learningModel.performanceHistory.length,
            currentDay: state.currentDay,
            totalTasksCompleted: state.completedTasks.size,
            averageStudyTime: state.learningModel.preferences.averageStudyTime
        },
        performanceHistory: state.learningModel.performanceHistory,
        categoryStrengths: state.learningModel.categoryStrengths,
        categoryWeaknesses: state.learningModel.categoryWeaknesses,
        preferences: state.learningModel.preferences,
        adaptations: state.learningModel.adaptations,
        completedTasks: Array.from(state.completedTasks),
        customTopics: state.customTopics
    };

    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `google-ml-prep-analytics-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    alert('📥 Learning data exported successfully! Use this to track your progress over time or share with mentors.');
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', init);

