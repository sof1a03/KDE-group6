import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { HttpParams } from '@angular/common/http';

export interface Book {
  name: string;
  url: string;
  id: string;
  genres: string[];
  publisher: string;
  year: number;
  ISBN: string;
  author: string;
}


@Injectable({
  providedIn: 'root'
})
export class BookService {
  private apiUrl = 'api/books'; // Replace with your actual API endpoint

  constructor(private http: HttpClient) { }

  // Should return paged results of size 9
  getBooks(): Observable<Book[]> {
    return this.http.get<Book[]>(this.apiUrl);
  }

  // Should return individual book
  getBook(id: string): Observable<Book> {
    const url = `api/books/${id}`;
    return this.http.get<Book>(url);
  }

  // Query for more books like the book with ID
  getMoreLike(id: string, page?: number, pagesize?: number): Observable<Book[]>{
    // Dummy code, replace later with proper API
    if (page !== undefined && pagesize !== undefined ){
      const params = new HttpParams()
      .set('page', page.toString())
      .set('pageSize', pagesize.toString());
      return this.http.get<Book[]>(this.apiUrl, {params});
    }
      return this.http.get<Book[]>(this.apiUrl);
  }
}
