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
        // GET - Fetch all notes
        if (event.httpMethod === 'GET') {
            const { data, error } = await supabase
                .from('review_notes')
                .select('*')
                .eq('user_id', userId)
                .order('created_at', { ascending: false });

            if (error) return handleSupabaseError(error);

            return successResponse(data);
        }

        // POST - Create new note
        if (event.httpMethod === 'POST') {
            const payload = JSON.parse(event.body || '{}');
            const { taskId, day, category, taskTitle, content } = payload;

            if (!taskId || !content || !day || !category) {
                return errorResponse(400, 'Missing required fields: taskId, day, category, content');
            }

            const { data, error } = await supabase
                .from('review_notes')
                .insert([{
                    user_id: userId,
                    task_id: taskId,
                    day: parseInt(day),
                    category,
                    task_title: taskTitle || '',
                    content: content.trim()
                }])
                .select()
                .single();

            if (error) return handleSupabaseError(error);

            return {
                ...successResponse(data),
                statusCode: 201
            };
        }

        // DELETE - Delete a note
        if (event.httpMethod === 'DELETE') {
            const noteId = event.queryStringParameters?.id;

            if (!noteId) {
                return errorResponse(400, 'Missing note id');
            }

            const { error } = await supabase
                .from('review_notes')
                .delete()
                .eq('id', parseInt(noteId))
                .eq('user_id', userId);

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
        console.error('Notes function error:', error);
        return errorResponse(500, 'Internal server error');
    }
};
