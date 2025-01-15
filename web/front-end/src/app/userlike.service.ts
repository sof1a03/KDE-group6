import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class UserlikeService {


  private apiUrl = 'localhost/likes'; // Replace with your actual API endpoint

  constructor(private http: HttpClient) { }

  // Should return paged results of size 9
  getLikes(userid: string): Observable<string[]> {
    const url = `api/likes/${userid}`;
    return this.http.get<string[]>(url);
  }

  likeBook(bookid: string, userid: string): Observable<string> {
    const url = `api/likes/${bookid}`;
    return this.http.post<string>(url,
      {userid: userid,
        bookid: bookid
      }
    );
  }
}

