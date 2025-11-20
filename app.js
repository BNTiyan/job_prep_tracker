// Application State
let state = {
    currentDay: 1,
    completedTasks: new Set(),
    customTopics: [],
    startDate: new Date().toISOString().split('T')[0],
    practiceProblems: null // Will load from YAML
};

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
            ...parsed,
            completedTasks: new Set(parsed.completedTasks || [])
        };
    }
}

// Save state to localStorage
function saveState() {
    localStorage.setItem('googlePrepState', JSON.stringify({
        ...state,
        completedTasks: Array.from(state.completedTasks)
    }));
}

// Initialize app
async function init() {
    await loadPracticeProblems();
    loadState();
    renderTasks();
    updateProgress();
    updateDateDisplay();
    setupEventListeners();
}

// Render tasks for current day
function renderTasks() {
    const container = document.getElementById('tasksContainer');
    container.innerHTML = '';

    // Get tasks for current day
    const dayPlan = preparationPlan.find(p => p.day === state.currentDay);
    
    if (!dayPlan) {
        container.innerHTML = '<div class="task-category"><p>No tasks planned for this day yet. Add custom topics!</p></div>';
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
    
    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    const dateString = date.toLocaleDateString('en-US', options);
    
    document.getElementById('currentDate').textContent = `Day ${state.currentDay} - ${dateString}`;
    
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
            saveState();
            renderTasks();
            updateProgress();
            updateDateDisplay();
        }
    });

    document.getElementById('nextDay').addEventListener('click', () => {
        if (state.currentDay < 60) {
            state.currentDay++;
            saveState();
            renderTasks();
            updateProgress();
            updateDateDisplay();
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
                startDate: new Date().toISOString().split('T')[0]
            };
            saveState();
            renderTasks();
            updateProgress();
            updateDateDisplay();
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

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', init);

