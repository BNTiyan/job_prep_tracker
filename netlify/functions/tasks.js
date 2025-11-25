import { supabase, successResponse, errorResponse, handleSupabaseError } from './supabase-client.js';

export const handler = async (event) => {
    // Handle OPTIONS preflight request
    if (event.httpMethod === 'OPTIONS') {
        return {
            statusCode: 204,
            headers: {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET,POST,DELETE,OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        };
    }

    // Check if Supabase is configured
    if (!supabase) {
        return errorResponse(500, 'Database not configured');
    }

    const userId = 'default_user'; // You can add auth later

    try {
        // GET - Fetch all completed tasks
        if (event.httpMethod === 'GET') {
            const { data, error } = await supabase
                .from('completed_tasks')
                .select('task_id')
                .eq('user_id', userId);

            if (error) return handleSupabaseError(error);

            const completedTaskIds = data.map(row => row.task_id);
            return successResponse({ completedTaskIds });
        }

        // POST - Mark task as completed or update status
        if (event.httpMethod === 'POST') {
            const payload = JSON.parse(event.body || '{}');
            const { taskId, day, category, taskTitle, completed = true } = payload;

            if (!taskId) {
                return errorResponse(400, 'Missing taskId');
            }

            if (completed) {
                // Insert or update completed task
                const { data, error } = await supabase
                    .from('completed_tasks')
                    .upsert({
                        user_id: userId,
                        task_id: taskId,
                        day: parseInt(day) || 0,
                        category: category || 'unknown',
                        task_title: taskTitle || '',
                        completed_at: new Date().toISOString()
                    }, {
                        onConflict: 'user_id,task_id'
                    })
                    .select()
                    .single();

                if (error) return handleSupabaseError(error);

                return successResponse({ taskId, completed: true, data });
            } else {
                // Remove completed status
                const { error } = await supabase
                    .from('completed_tasks')
                    .delete()
                    .eq('user_id', userId)
                    .eq('task_id', taskId);

                if (error) return handleSupabaseError(error);

                return successResponse({ taskId, completed: false });
            }
        }

        // DELETE - Remove completed task
        if (event.httpMethod === 'DELETE') {
            const taskId = event.queryStringParameters?.taskId;

            if (!taskId) {
                return errorResponse(400, 'Missing taskId');
            }

            const { error } = await supabase
                .from('completed_tasks')
                .delete()
                .eq('user_id', userId)
                .eq('task_id', taskId);

            if (error) return handleSupabaseError(error);

            return {
                statusCode: 204,
                headers: {
                    'Access-Control-Allow-Origin': '*'
                },
                body: ''
            };
        }

        return errorResponse(405, 'Method not allowed');

    } catch (error) {
        console.error('Tasks function error:', error);
        return errorResponse(500, 'Internal server error');
    }
};
