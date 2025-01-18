import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class UserlikeService {

  private userLikes: { [username: string]: { [bookId: string]: string} } = {};

  constructor(private http: HttpClient) {
    const storedLikes = localStorage.getItem('userLikes');
    if (storedLikes) {
      this.userLikes = JSON.parse(storedLikes);
    }
  }

  like(username: string, bookId: string, bookTitle: string): void {
    if (!this.userLikes[username]) {
      this.userLikes[username] = {};
    }
    this.userLikes[username][bookId] = bookTitle;
    localStorage.setItem('userLikes', JSON.stringify(this.userLikes));
    console.log(this.userLikes)
  }

  hasLiked(username: string, bookId: string): boolean {
    return this.userLikes[username] && this.userLikes[username][bookId] !== "";
  }

  getLikedBooks(username: string): { bookId: string, bookTitle: string }[] {
    if (this.userLikes[username]) {
      return Object.entries(this.userLikes[username]).map(([bookId, bookTitle]) => ({ bookId, bookTitle }));
    }
    return [];
  }

  removeLike(username: string, bookId: string): void {
    if (this.userLikes[username]) {
      delete this.userLikes[username][bookId];
      localStorage.setItem('userLikes', JSON.stringify(this.userLikes));
    }
}
}
