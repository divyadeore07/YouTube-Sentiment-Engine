from datetime import datetime
import yt_dlp

def fetch_youtube_comments(video_url: str, max_comments: int = 300):
    """
    Extracts raw comment strings, authors, likes, and publication timestamps 
    from a given YouTube video URL using mobile client emulation to bypass bot checks.
    """
    ydl_opts = {
        'getcomments': True,
        'extract_flat': False,
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
        'extractor_args': {
            'youtube': {
                'max_comments': [str(max_comments)],
                'player_client': ['android', 'ios', 'web_creator'],
                'comment_sort': ['top']
            }
        }
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            raw_comments = info.get('comments', [])
            
            parsed_comments = []
            for c in raw_comments:
                ts = c.get('timestamp')
                
                if ts:
                    try:
                        date_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    except Exception:
                        date_str = datetime.now().strftime('%Y-%m-%d')
                else:
                    date_str = datetime.now().strftime('%Y-%m-%d')
                
                parsed_comments.append({
                    'author': c.get('author', 'Anonymous'),
                    'text': c.get('text', ''),
                    'like_count': c.get('like_count', 0),
                    'timestamp': date_str
                })
            
            video_title = info.get('title', 'YouTube Video')
            return video_title, parsed_comments

    except Exception as e:
        raise Exception(f"Failed to fetch comments: {str(e)}")